"""
Static code analysis service for extracting AST-based insights.
Provides structural understanding of code changes before LLM review.
"""
import ast
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from app.models.schemas import CodeChange, ReviewCategory, Severity
from app.core.logging import logger


@dataclass
class CodeMetrics:
    """Metrics extracted from code analysis."""
    cyclomatic_complexity: int
    function_count: int
    class_count: int
    line_count: int
    docstring_coverage: float
    import_count: int
    has_type_hints: bool
    has_error_handling: bool


class CodeAnalyzer:
    """Analyzes Python code for structural patterns and issues."""
    
    def __init__(self):
        self.logger = logger.bind(service="code_analyzer")
    
    def analyze_change(self, change: CodeChange) -> Dict[str, any]:
        self.logger.info("analyzing_code_change", file=change.filename)
        code = self._extract_code_from_patch(change.patch) if change.patch else change.new_content
        if not code:
            return {"error": "No code content available"}
        
        results = {
            "file_path": change.filename,
            "language": self._detect_language(change.filename),
            "metrics": None,
            "ast_issues": [],
            "security_flags": [],
            "style_flags": []
        }
        
        if results["language"] == "python":
            try:
                tree = ast.parse(code)
                results["metrics"] = self._calculate_metrics(tree, code)
                results["ast_issues"] = self._detect_ast_issues(tree)
                results["security_flags"] = self._detect_security_issues(tree, code)
                results["style_flags"] = self._detect_style_issues(tree, code)
            except SyntaxError as e:
                self.logger.warning("syntax_error", file=change.filename, error=str(e))
                results["ast_issues"].append({
                    "category": ReviewCategory.FUNCTIONAL,
                    "severity": Severity.HIGH,
                    "message": f"Syntax error in file: {str(e)}"
                })
        return results
    
    def _extract_code_from_patch(self, patch: str) -> str:
        lines = []
        for line in patch.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                lines.append(line[1:])
            elif not line.startswith("-") and not line.startswith("@@") and not line.startswith("---"):
                lines.append(line)
        return "\n".join(lines)
    
    def _detect_language(self, filename: str) -> str:
        extension_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".cpp": "cpp", ".c": "c", ".go": "go",
            ".rs": "rust", ".rb": "ruby"
        }
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        return extension_map.get(ext, "unknown")
    
    def _calculate_metrics(self, tree: ast.AST, code: str) -> CodeMetrics:
        lines = code.split("\n")
        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.Assert, ast.comprehension)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        has_type_hints = any(
            isinstance(node, ast.AnnAssign) or 
            (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and 
             (node.returns or any(arg.annotation for arg in node.args.args)))
            for node in ast.walk(tree)
        )
        
        has_error_handling = any(isinstance(node, ast.Try) for node in ast.walk(tree))
        
        documented = 0
        total = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total += 1
                if ast.get_docstring(node):
                    documented += 1
        
        docstring_coverage = (documented / total * 100) if total > 0 else 0
        
        return CodeMetrics(
            cyclomatic_complexity=complexity,
            function_count=len(functions),
            class_count=len(classes),
            line_count=len(lines),
            docstring_coverage=docstring_coverage,
            import_count=len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
            has_type_hints=has_type_hints,
            has_error_handling=has_error_handling
        )
    
    def _detect_ast_issues(self, tree: ast.AST) -> List[Dict]:
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "category": ReviewCategory.FUNCTIONAL,
                        "severity": Severity.HIGH,
                        "message": "Bare 'except:' clause detected. Use 'except Exception:' or specific exceptions.",
                        "line": node.lineno
                    })
            
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append({
                            "category": ReviewCategory.FUNCTIONAL,
                            "severity": Severity.HIGH,
                            "message": "Mutable default argument detected. Use None as default and initialize inside function.",
                            "line": node.lineno
                        })
        return issues
    
    def _detect_security_issues(self, tree: ast.AST, code: str) -> List[Dict]:
        issues = []
        
        dangerous_funcs = ["eval(", "exec(", "os.system(", "pickle.loads", "pickle.load("]
        for func in dangerous_funcs:
            if func in code:
                msg_map = {
                    "eval(": "Use of eval() detected - potential code injection risk",
                    "exec(": "Use of exec() detected - potential code injection risk",
                    "os.system(": "Use of os.system() detected - consider subprocess.run()",
                    "pickle.loads": "Unsafe deserialization with pickle",
                    "pickle.load(": "Unsafe deserialization with pickle"
                }
                issues.append({
                    "category": ReviewCategory.SECURITY,
                    "severity": Severity.CRITICAL,
                    "message": msg_map.get(func, "Dangerous function detected")
                })
        
        if "shell=True" in code and "subprocess" in code:
            issues.append({
                "category": ReviewCategory.SECURITY,
                "severity": Severity.CRITICAL,
                "message": "shell=True in subprocess - injection risk"
            })
        
        if "yaml.load(" in code and "yaml.safe_load" not in code:
            issues.append({
                "category": ReviewCategory.SECURITY,
                "severity": Severity.CRITICAL,
                "message": "Unsafe yaml.load() - use yaml.safe_load()"
            })
        
        secret_keywords = ["password", "api_key", "secret", "token"]
        for keyword in secret_keywords:
            idx = code.lower().find(keyword + "=")
            if idx != -1:
                after = code[idx + len(keyword) + 1:].strip()
                if after.startswith('"') or after.startswith("'"):
                    quote = after[0]
                    end_idx = after.find(quote, 1)
                    if end_idx != -1:
                        value = after[1:end_idx]
                        skip = False
                        for p in ["placeholder", "example", "dummy", "test", "xxx", "<", ">"]:
                            if p in value.lower():
                                skip = True
                                break
                        if not skip:
                            msg = "Possible hardcoded " + keyword + ": " + value[:30] + "..."
                            issues.append({
                                "category": ReviewCategory.SECURITY,
                                "severity": Severity.CRITICAL,
                                "message": msg
                            })
        return issues
    
    def _detect_style_issues(self, tree: ast.AST, code: str) -> List[Dict]:
        issues = []
        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > 120:
                issues.append({
                    "category": ReviewCategory.STYLE,
                    "severity": Severity.LOW,
                    "message": f"Line exceeds 120 characters ({len(line)} chars)",
                    "line": i
                })
            
            if re.search(r"#\s*(TODO|FIXME|HACK|XXX|BUG)", line, re.IGNORECASE):
                if not re.search(r"#\d+", line):
                    issues.append({
                        "category": ReviewCategory.DOCUMENTATION,
                        "severity": Severity.LOW,
                        "message": f"TODO/FIXME comment without issue reference at line {i}",
                        "line": i
                    })
        return issues


# Singleton instance
code_analyzer = CodeAnalyzer()