/**
 * Code Analysis Service
 * Provides deep code understanding through AST parsing and semantic analysis
 * 
 * Features:
 * - AST parsing with ts-morph for TypeScript/JavaScript
 * - Function/class/variable extraction
 * - Dependency graph building
 * - Call graph analysis
 * - Code complexity metrics
 * - Import/export resolution
 * - Design pattern detection
 * - Security vulnerability scanning
 * - Code smell detection
 */

import { Project, SourceFile, Node, SyntaxKind, FunctionDeclaration, ClassDeclaration, VariableDeclaration, ImportDeclaration, CallExpression, ArrowFunction, MethodDeclaration, PropertyAccessExpression, VariableStatement, InterfaceDeclaration, TypeAliasDeclaration, EnumDeclaration } from 'ts-morph';

// Code entity types for analysis
export interface CodeEntity {
  name: string;
  type: 'function' | 'class' | 'variable' | 'interface' | 'type' | 'enum' | 'method' | 'property' | 'arrow-function';
  filePath: string;
  startLine: number;
  endLine: number;
  signature?: string;
  docComment?: string;
  complexity?: number;
  dependencies: string[];
  calledBy: string[];
  calls: string[];
  parameters?: ParameterInfo[];
  returnType?: string;
  modifiers?: string[];
  isAsync?: boolean;
  isExported?: boolean;
}

export interface ParameterInfo {
  name: string;
  type: string;
  isOptional: boolean;
  defaultValue?: string;
}

export interface ImportInfo {
  moduleSpecifier: string;
  namedImports: string[];
  defaultImport?: string;
  namespaceImport?: string;
  isExternal: boolean;
  filePath: string;
  line: number;
}

export interface ExportInfo {
  name: string;
  type: 'function' | 'class' | 'variable' | 'interface' | 'type' | 'default' | 'namespace';
  filePath: string;
  line: number;
}

export interface DependencyNode {
  filePath: string;
  imports: ImportInfo[];
  exports: ExportInfo[];
  entities: CodeEntity[];
  incomingEdges: string[];
  outgoingEdges: string[];
}

export interface CallGraphNode {
  name: string;
  filePath: string;
  calls: { name: string; filePath: string; line: number }[];
  calledBy: { name: string; filePath: string; line: number }[];
}

export interface CodeAnalysisResult {
  entities: CodeEntity[];
  imports: ImportInfo[];
  exports: ExportInfo[];
  complexity: ComplexityMetrics;
  issues: CodeIssue[];
  patterns: DetectedPattern[];
  securityIssues: SecurityIssue[];
  codeSmells: CodeSmell[];
  summary: AnalysisSummary;
}

export interface ComplexityMetrics {
  cyclomaticComplexity: number;
  cognitiveComplexity: number;
  linesOfCode: number;
  linesOfComments: number;
  functionCount: number;
  classCount: number;
  avgFunctionLength: number;
  maxFunctionLength: number;
  maintainabilityIndex: number;
}

export interface CodeIssue {
  type: 'error' | 'warning' | 'info';
  category: 'syntax' | 'type' | 'logic' | 'style' | 'performance' | 'security';
  message: string;
  filePath: string;
  line: number;
  column?: number;
  code?: string;
  suggestion?: string;
}

export interface DetectedPattern {
  name: string;
  type: 'creational' | 'structural' | 'behavioral' | 'architectural';
  confidence: number;
  location: { filePath: string; startLine: number; endLine: number };
  description: string;
}

export interface SecurityIssue {
  severity: 'critical' | 'high' | 'medium' | 'low';
  type: string;
  message: string;
  filePath: string;
  line: number;
  cwe?: string;
  recommendation: string;
}

export interface CodeSmell {
  type: string;
  severity: 'major' | 'minor' | 'info';
  message: string;
  filePath: string;
  line: number;
  suggestion: string;
}

export interface AnalysisSummary {
  totalFiles: number;
  totalLines: number;
  totalFunctions: number;
  totalClasses: number;
  averageComplexity: number;
  healthScore: number;
  topIssues: string[];
}

export interface DependencyGraph {
  nodes: Map<string, DependencyNode>;
  edges: Array<{ from: string; to: string; type: 'import' | 'export' | 'call' }>;
}

export interface FunctionContext {
  name: string;
  parameters: ParameterInfo[];
  returnType: string;
  dependencies: string[];
  sideEffects: string[];
  complexity: number;
  purpose: string;
}

export interface RefactoringOpportunity {
  type: 'extract-function' | 'extract-variable' | 'rename' | 'inline' | 'move' | 'simplify-conditional';
  description: string;
  filePath: string;
  startLine: number;
  endLine: number;
  suggestedCode?: string;
  impact: 'low' | 'medium' | 'high';
}

// Security patterns to detect
const SECURITY_PATTERNS = {
  sqlInjection: /(\$\{.*\}|`.*\$\{.*\}.*`)\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)/i,
  xss: /(innerHTML|outerHTML|document\.write)\s*=\s*[^"'`]/,
  evalUsage: /\beval\s*\(/,
  dangerousRegex: /new\s+RegExp\s*\(\s*[^"'`]/,
  hardcodedSecrets: /(password|secret|api[_-]?key|token|credential)\s*[=:]\s*['"][^'"]{8,}['"]/i,
  unsafeDeserialize: /JSON\.parse\s*\(\s*[^)]*\)/,
  commandInjection: /exec\s*\(|spawn\s*\(|execSync\s*\(/,
  pathTraversal: /\.\.\//,
  prototypePolluton: /__proto__|constructor\s*\[/,
};

// Code smell patterns
const CODE_SMELL_PATTERNS = {
  longFunction: { threshold: 50, message: 'Function exceeds recommended length' },
  deepNesting: { threshold: 4, message: 'Excessive nesting depth' },
  tooManyParameters: { threshold: 5, message: 'Too many function parameters' },
  longFile: { threshold: 500, message: 'File exceeds recommended length' },
  duplicateCode: { threshold: 10, message: 'Potential code duplication' },
  complexCondition: { threshold: 3, message: 'Complex conditional expression' },
  magicNumbers: { pattern: /[^a-zA-Z0-9_]([2-9]\d{2,}|[1-9]\d{3,})[^a-zA-Z0-9_]/, message: 'Magic number detected' },
  todoComments: { pattern: /\/\/\s*(TODO|FIXME|HACK|XXX)/i, message: 'Unresolved TODO/FIXME comment' },
};

export class CodeAnalyzer {
  private project: Project;
  private entityCache: Map<string, CodeEntity[]> = new Map();
  private dependencyGraph: DependencyGraph = { nodes: new Map(), edges: [] };
  private callGraph: Map<string, CallGraphNode> = new Map();

  constructor() {
    this.project = new Project({
      useInMemoryFileSystem: true,
      compilerOptions: {
        target: 99, // ESNext
        module: 99, // ESNext
        moduleResolution: 2, // Node
        strict: true,
        esModuleInterop: true,
        skipLibCheck: true,
        allowJs: true,
        jsx: 4, // ReactJSX
      },
    });
  }

  /**
   * Analyze a single file and extract all code entities
   */
  async analyzeFile(filePath: string, content: string): Promise<CodeAnalysisResult> {
    const sourceFile = this.project.createSourceFile(filePath, content, { overwrite: true });
    
    const entities = this.extractEntities(sourceFile, filePath);
    const imports = this.extractImports(sourceFile, filePath);
    const exports = this.extractExports(sourceFile, filePath);
    const complexity = this.calculateComplexity(sourceFile);
    const issues = this.detectIssues(sourceFile, filePath);
    const patterns = this.detectPatterns(sourceFile, filePath);
    const securityIssues = this.detectSecurityIssues(content, filePath);
    const codeSmells = this.detectCodeSmells(sourceFile, filePath, content);
    
    // Cache entities for later use
    this.entityCache.set(filePath, entities);
    
    // Update dependency graph
    this.updateDependencyGraph(filePath, imports, exports, entities);
    
    // Build call graph
    this.buildCallGraph(sourceFile, filePath);

    const summary = this.generateSummary(entities, complexity, issues, securityIssues, codeSmells);
    
    return {
      entities,
      imports,
      exports,
      complexity,
      issues,
      patterns,
      securityIssues,
      codeSmells,
      summary,
    };
  }

  /**
   * Analyze multiple files and build a complete project understanding
   */
  async analyzeProject(files: Array<{ path: string; content: string }>): Promise<{
    files: Map<string, CodeAnalysisResult>;
    dependencyGraph: DependencyGraph;
    callGraph: Map<string, CallGraphNode>;
    projectSummary: AnalysisSummary;
    refactoringOpportunities: RefactoringOpportunity[];
  }> {
    const fileResults = new Map<string, CodeAnalysisResult>();
    
    // First pass: analyze all files
    for (const file of files) {
      const result = await this.analyzeFile(file.path, file.content);
      fileResults.set(file.path, result);
    }
    
    // Second pass: resolve cross-file references
    this.resolveCrossFileReferences();
    
    // Generate project-level summary
    const projectSummary = this.generateProjectSummary(fileResults);
    
    // Detect refactoring opportunities
    const refactoringOpportunities = this.detectRefactoringOpportunities(fileResults);
    
    return {
      files: fileResults,
      dependencyGraph: this.dependencyGraph,
      callGraph: this.callGraph,
      projectSummary,
      refactoringOpportunities,
    };
  }

  /**
   * Extract all code entities from a source file
   */
  private extractEntities(sourceFile: SourceFile, filePath: string): CodeEntity[] {
    const entities: CodeEntity[] = [];
    
    // Extract functions
    sourceFile.getFunctions().forEach(func => {
      entities.push(this.extractFunctionEntity(func, filePath));
    });
    
    // Extract classes
    sourceFile.getClasses().forEach(cls => {
      entities.push(this.extractClassEntity(cls, filePath));
      
      // Extract methods from classes
      cls.getMethods().forEach(method => {
        entities.push(this.extractMethodEntity(method, filePath, cls.getName() || 'Anonymous'));
      });
    });
    
    // Extract variable declarations (including arrow functions)
    sourceFile.getVariableStatements().forEach(statement => {
      statement.getDeclarations().forEach(decl => {
        const initializer = decl.getInitializer();
        if (initializer && Node.isArrowFunction(initializer)) {
          entities.push(this.extractArrowFunctionEntity(decl, initializer, filePath));
        } else {
          entities.push(this.extractVariableEntity(decl, filePath));
        }
      });
    });
    
    // Extract interfaces
    sourceFile.getInterfaces().forEach(iface => {
      entities.push(this.extractInterfaceEntity(iface, filePath));
    });
    
    // Extract type aliases
    sourceFile.getTypeAliases().forEach(typeAlias => {
      entities.push(this.extractTypeAliasEntity(typeAlias, filePath));
    });
    
    // Extract enums
    sourceFile.getEnums().forEach(enumDecl => {
      entities.push(this.extractEnumEntity(enumDecl, filePath));
    });
    
    return entities;
  }

  private extractFunctionEntity(func: FunctionDeclaration, filePath: string): CodeEntity {
    const name = func.getName() || 'anonymous';
    const params = func.getParameters().map(p => ({
      name: p.getName(),
      type: p.getType().getText(),
      isOptional: p.isOptional(),
      defaultValue: p.getInitializer()?.getText(),
    }));
    
    return {
      name,
      type: 'function',
      filePath,
      startLine: func.getStartLineNumber(),
      endLine: func.getEndLineNumber(),
      signature: this.getFunctionSignature(func),
      docComment: func.getJsDocs().map(d => d.getDescription()).join('\n'),
      complexity: this.calculateFunctionComplexity(func),
      dependencies: this.extractDependencies(func),
      calledBy: [],
      calls: this.extractCalls(func),
      parameters: params,
      returnType: func.getReturnType().getText(),
      modifiers: func.getModifiers().map(m => m.getText()),
      isAsync: func.isAsync(),
      isExported: func.isExported(),
    };
  }

  private extractClassEntity(cls: ClassDeclaration, filePath: string): CodeEntity {
    return {
      name: cls.getName() || 'Anonymous',
      type: 'class',
      filePath,
      startLine: cls.getStartLineNumber(),
      endLine: cls.getEndLineNumber(),
      signature: this.getClassSignature(cls),
      docComment: cls.getJsDocs().map(d => d.getDescription()).join('\n'),
      complexity: this.calculateClassComplexity(cls),
      dependencies: this.extractClassDependencies(cls),
      calledBy: [],
      calls: [],
      modifiers: cls.getModifiers().map(m => m.getText()),
      isExported: cls.isExported(),
    };
  }

  private extractMethodEntity(method: MethodDeclaration, filePath: string, className: string): CodeEntity {
    const params = method.getParameters().map(p => ({
      name: p.getName(),
      type: p.getType().getText(),
      isOptional: p.isOptional(),
      defaultValue: p.getInitializer()?.getText(),
    }));
    
    return {
      name: `${className}.${method.getName()}`,
      type: 'method',
      filePath,
      startLine: method.getStartLineNumber(),
      endLine: method.getEndLineNumber(),
      signature: this.getMethodSignature(method),
      docComment: method.getJsDocs().map(d => d.getDescription()).join('\n'),
      complexity: this.calculateMethodComplexity(method),
      dependencies: this.extractDependencies(method),
      calledBy: [],
      calls: this.extractCalls(method),
      parameters: params,
      returnType: method.getReturnType().getText(),
      modifiers: method.getModifiers().map(m => m.getText()),
      isAsync: method.isAsync(),
    };
  }

  private extractArrowFunctionEntity(decl: VariableDeclaration, arrow: ArrowFunction, filePath: string): CodeEntity {
    const params = arrow.getParameters().map(p => ({
      name: p.getName(),
      type: p.getType().getText(),
      isOptional: p.isOptional(),
      defaultValue: p.getInitializer()?.getText(),
    }));
    
    return {
      name: decl.getName(),
      type: 'arrow-function',
      filePath,
      startLine: decl.getStartLineNumber(),
      endLine: decl.getEndLineNumber(),
      signature: `const ${decl.getName()} = ${this.getArrowSignature(arrow)}`,
      complexity: this.calculateArrowComplexity(arrow),
      dependencies: this.extractArrowDependencies(arrow),
      calledBy: [],
      calls: this.extractArrowCalls(arrow),
      parameters: params,
      returnType: arrow.getReturnType().getText(),
      isAsync: arrow.isAsync(),
      isExported: decl.getVariableStatement()?.isExported() || false,
    };
  }

  private extractVariableEntity(decl: VariableDeclaration, filePath: string): CodeEntity {
    return {
      name: decl.getName(),
      type: 'variable',
      filePath,
      startLine: decl.getStartLineNumber(),
      endLine: decl.getEndLineNumber(),
      signature: `${decl.getVariableStatement()?.getDeclarationKind()} ${decl.getName()}: ${decl.getType().getText()}`,
      dependencies: [],
      calledBy: [],
      calls: [],
      isExported: decl.getVariableStatement()?.isExported() || false,
    };
  }

  private extractInterfaceEntity(iface: InterfaceDeclaration, filePath: string): CodeEntity {
    return {
      name: iface.getName(),
      type: 'interface',
      filePath,
      startLine: iface.getStartLineNumber(),
      endLine: iface.getEndLineNumber(),
      signature: this.getInterfaceSignature(iface),
      docComment: iface.getJsDocs().map(d => d.getDescription()).join('\n'),
      dependencies: this.extractInterfaceDependencies(iface),
      calledBy: [],
      calls: [],
      isExported: iface.isExported(),
    };
  }

  private extractTypeAliasEntity(typeAlias: TypeAliasDeclaration, filePath: string): CodeEntity {
    return {
      name: typeAlias.getName(),
      type: 'type',
      filePath,
      startLine: typeAlias.getStartLineNumber(),
      endLine: typeAlias.getEndLineNumber(),
      signature: `type ${typeAlias.getName()} = ${typeAlias.getType().getText()}`,
      docComment: typeAlias.getJsDocs().map(d => d.getDescription()).join('\n'),
      dependencies: [],
      calledBy: [],
      calls: [],
      isExported: typeAlias.isExported(),
    };
  }

  private extractEnumEntity(enumDecl: EnumDeclaration, filePath: string): CodeEntity {
    return {
      name: enumDecl.getName(),
      type: 'enum',
      filePath,
      startLine: enumDecl.getStartLineNumber(),
      endLine: enumDecl.getEndLineNumber(),
      signature: `enum ${enumDecl.getName()} { ${enumDecl.getMembers().map(m => m.getName()).join(', ')} }`,
      docComment: enumDecl.getJsDocs().map(d => d.getDescription()).join('\n'),
      dependencies: [],
      calledBy: [],
      calls: [],
      isExported: enumDecl.isExported(),
    };
  }

  /**
   * Extract imports from a source file
   */
  private extractImports(sourceFile: SourceFile, filePath: string): ImportInfo[] {
    return sourceFile.getImportDeclarations().map(imp => {
      const moduleSpecifier = imp.getModuleSpecifierValue();
      const namedImports = imp.getNamedImports().map(n => n.getName());
      const defaultImport = imp.getDefaultImport()?.getText();
      const namespaceImport = imp.getNamespaceImport()?.getText();
      
      return {
        moduleSpecifier,
        namedImports,
        defaultImport,
        namespaceImport,
        isExternal: !moduleSpecifier.startsWith('.') && !moduleSpecifier.startsWith('/'),
        filePath,
        line: imp.getStartLineNumber(),
      };
    });
  }

  /**
   * Extract exports from a source file
   */
  private extractExports(sourceFile: SourceFile, filePath: string): ExportInfo[] {
    const exports: ExportInfo[] = [];
    
    // Named exports
    sourceFile.getExportDeclarations().forEach(exp => {
      exp.getNamedExports().forEach(named => {
        exports.push({
          name: named.getName(),
          type: 'variable',
          filePath,
          line: exp.getStartLineNumber(),
        });
      });
    });
    
    // Exported declarations
    sourceFile.getExportedDeclarations().forEach((declarations, name) => {
      declarations.forEach(decl => {
        let type: ExportInfo['type'] = 'variable';
        if (Node.isFunctionDeclaration(decl)) type = 'function';
        else if (Node.isClassDeclaration(decl)) type = 'class';
        else if (Node.isInterfaceDeclaration(decl)) type = 'interface';
        else if (Node.isTypeAliasDeclaration(decl)) type = 'type';
        
        exports.push({
          name,
          type,
          filePath,
          line: decl.getStartLineNumber(),
        });
      });
    });
    
    return exports;
  }

  /**
   * Calculate complexity metrics for a source file
   */
  private calculateComplexity(sourceFile: SourceFile): ComplexityMetrics {
    let cyclomaticComplexity = 1;
    let cognitiveComplexity = 0;
    let functionCount = 0;
    let classCount = 0;
    let totalFunctionLength = 0;
    let maxFunctionLength = 0;
    
    const text = sourceFile.getFullText();
    const lines = text.split('\n');
    const linesOfCode = lines.filter(l => l.trim().length > 0).length;
    const linesOfComments = lines.filter(l => l.trim().startsWith('//') || l.trim().startsWith('/*') || l.trim().startsWith('*')).length;
    
    // Count decision points for cyclomatic complexity
    sourceFile.forEachDescendant(node => {
      const kind = node.getKind();
      
      // Cyclomatic complexity: count branches
      if ([SyntaxKind.IfStatement, SyntaxKind.ConditionalExpression, SyntaxKind.CaseClause,
           SyntaxKind.ForStatement, SyntaxKind.ForInStatement, SyntaxKind.ForOfStatement,
           SyntaxKind.WhileStatement, SyntaxKind.DoStatement, SyntaxKind.CatchClause].includes(kind)) {
        cyclomaticComplexity++;
      }
      
      // Logical operators add to complexity
      if ([SyntaxKind.AmpersandAmpersandToken, SyntaxKind.BarBarToken, SyntaxKind.QuestionQuestionToken].includes(kind)) {
        cyclomaticComplexity++;
      }
      
      // Cognitive complexity: account for nesting
      if (Node.isIfStatement(node) || Node.isForStatement(node) || Node.isWhileStatement(node)) {
        const depth = this.getNestingDepth(node);
        cognitiveComplexity += 1 + depth;
      }
      
      // Count functions
      if (Node.isFunctionDeclaration(node) || Node.isArrowFunction(node) || Node.isFunctionExpression(node)) {
        functionCount++;
        const funcLines = node.getEndLineNumber() - node.getStartLineNumber() + 1;
        totalFunctionLength += funcLines;
        maxFunctionLength = Math.max(maxFunctionLength, funcLines);
      }
      
      // Count classes
      if (Node.isClassDeclaration(node)) {
        classCount++;
      }
    });
    
    const avgFunctionLength = functionCount > 0 ? totalFunctionLength / functionCount : 0;
    
    // Maintainability index (simplified version of the Halstead-based formula)
    const maintainabilityIndex = Math.max(0, Math.min(100, 
      171 - 5.2 * Math.log(cyclomaticComplexity) - 0.23 * cognitiveComplexity - 16.2 * Math.log(linesOfCode)
    ));
    
    return {
      cyclomaticComplexity,
      cognitiveComplexity,
      linesOfCode,
      linesOfComments,
      functionCount,
      classCount,
      avgFunctionLength: Math.round(avgFunctionLength * 10) / 10,
      maxFunctionLength,
      maintainabilityIndex: Math.round(maintainabilityIndex),
    };
  }

  /**
   * Detect code issues (type errors, style issues, etc.)
   */
  private detectIssues(sourceFile: SourceFile, filePath: string): CodeIssue[] {
    const issues: CodeIssue[] = [];
    
    // Get TypeScript diagnostics
    const diagnostics = sourceFile.getPreEmitDiagnostics();
    for (const diagnostic of diagnostics) {
      const message = diagnostic.getMessageText();
      const messageText = typeof message === 'string' ? message : message.getMessageText();
      
      issues.push({
        type: diagnostic.getCategory() === 1 ? 'error' : 'warning',
        category: 'type',
        message: messageText,
        filePath,
        line: diagnostic.getLineNumber() || 1,
        code: `TS${diagnostic.getCode()}`,
      });
    }
    
    // Custom issue detection
    sourceFile.forEachDescendant(node => {
      // Detect any type usage
      if (Node.isTypeReference(node) && node.getText() === 'any') {
        issues.push({
          type: 'warning',
          category: 'type',
          message: 'Avoid using "any" type',
          filePath,
          line: node.getStartLineNumber(),
          suggestion: 'Use a more specific type or "unknown" if the type is truly unknown',
        });
      }
      
      // Detect console.log in production code
      if (Node.isCallExpression(node)) {
        const expr = node.getExpression();
        if (Node.isPropertyAccessExpression(expr)) {
          if (expr.getText().startsWith('console.')) {
            issues.push({
              type: 'info',
              category: 'style',
              message: 'Console statement detected',
              filePath,
              line: node.getStartLineNumber(),
              suggestion: 'Consider removing console statements or using a logging library',
            });
          }
        }
      }
      
      // Detect empty catch blocks
      if (Node.isCatchClause(node)) {
        const block = node.getBlock();
        if (block.getStatements().length === 0) {
          issues.push({
            type: 'warning',
            category: 'logic',
            message: 'Empty catch block',
            filePath,
            line: node.getStartLineNumber(),
            suggestion: 'Handle the error or at minimum log it',
          });
        }
      }
    });
    
    return issues;
  }

  /**
   * Detect design patterns in the code
   */
  private detectPatterns(sourceFile: SourceFile, filePath: string): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];
    
    sourceFile.getClasses().forEach(cls => {
      const className = cls.getName() || '';
      const methods = cls.getMethods();
      const properties = cls.getProperties();
      
      // Singleton pattern detection
      const hasPrivateConstructor = cls.getConstructors().some(c => 
        c.getModifiers().some(m => m.getText() === 'private')
      );
      const hasStaticInstance = properties.some(p => 
        p.isStatic() && p.getName().toLowerCase().includes('instance')
      );
      const hasGetInstance = methods.some(m => 
        m.isStatic() && m.getName().toLowerCase().includes('getinstance')
      );
      
      if (hasPrivateConstructor && (hasStaticInstance || hasGetInstance)) {
        patterns.push({
          name: 'Singleton',
          type: 'creational',
          confidence: 0.9,
          location: { filePath, startLine: cls.getStartLineNumber(), endLine: cls.getEndLineNumber() },
          description: 'Class implements the Singleton pattern with private constructor and static instance access',
        });
      }
      
      // Factory pattern detection
      if (className.toLowerCase().includes('factory') || 
          methods.some(m => m.getName().toLowerCase().startsWith('create'))) {
        patterns.push({
          name: 'Factory',
          type: 'creational',
          confidence: 0.7,
          location: { filePath, startLine: cls.getStartLineNumber(), endLine: cls.getEndLineNumber() },
          description: 'Class appears to implement the Factory pattern',
        });
      }
      
      // Observer pattern detection
      const hasSubscribe = methods.some(m => 
        ['subscribe', 'on', 'addListener', 'addEventListener'].includes(m.getName().toLowerCase())
      );
      const hasUnsubscribe = methods.some(m => 
        ['unsubscribe', 'off', 'removeListener', 'removeEventListener'].includes(m.getName().toLowerCase())
      );
      const hasNotify = methods.some(m => 
        ['notify', 'emit', 'dispatch', 'publish'].includes(m.getName().toLowerCase())
      );
      
      if (hasSubscribe && hasUnsubscribe && hasNotify) {
        patterns.push({
          name: 'Observer',
          type: 'behavioral',
          confidence: 0.85,
          location: { filePath, startLine: cls.getStartLineNumber(), endLine: cls.getEndLineNumber() },
          description: 'Class implements the Observer/Event Emitter pattern',
        });
      }
      
      // Strategy pattern detection
      const implementsInterface = cls.getImplements().length > 0;
      const hasExecuteOrRun = methods.some(m => 
        ['execute', 'run', 'process', 'handle'].includes(m.getName().toLowerCase())
      );
      
      if (implementsInterface && hasExecuteOrRun && methods.length <= 3) {
        patterns.push({
          name: 'Strategy',
          type: 'behavioral',
          confidence: 0.6,
          location: { filePath, startLine: cls.getStartLineNumber(), endLine: cls.getEndLineNumber() },
          description: 'Class may implement the Strategy pattern',
        });
      }
    });
    
    // React Hook pattern detection
    sourceFile.getFunctions().forEach(func => {
      const name = func.getName() || '';
      if (name.startsWith('use') && name.length > 3 && name[3] === name[3].toUpperCase()) {
        patterns.push({
          name: 'React Custom Hook',
          type: 'architectural',
          confidence: 0.95,
          location: { filePath, startLine: func.getStartLineNumber(), endLine: func.getEndLineNumber() },
          description: `Custom React hook: ${name}`,
        });
      }
    });
    
    return patterns;
  }

  /**
   * Detect security vulnerabilities
   */
  private detectSecurityIssues(content: string, filePath: string): SecurityIssue[] {
    const issues: SecurityIssue[] = [];
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      const lineNum = index + 1;
      
      // SQL Injection
      if (SECURITY_PATTERNS.sqlInjection.test(line)) {
        issues.push({
          severity: 'critical',
          type: 'SQL Injection',
          message: 'Potential SQL injection vulnerability detected',
          filePath,
          line: lineNum,
          cwe: 'CWE-89',
          recommendation: 'Use parameterized queries or an ORM instead of string concatenation',
        });
      }
      
      // XSS
      if (SECURITY_PATTERNS.xss.test(line)) {
        issues.push({
          severity: 'high',
          type: 'Cross-Site Scripting (XSS)',
          message: 'Potential XSS vulnerability - direct DOM manipulation',
          filePath,
          line: lineNum,
          cwe: 'CWE-79',
          recommendation: 'Use framework-provided sanitization or escape user input',
        });
      }
      
      // Eval usage
      if (SECURITY_PATTERNS.evalUsage.test(line)) {
        issues.push({
          severity: 'high',
          type: 'Code Injection',
          message: 'Use of eval() detected - potential code injection',
          filePath,
          line: lineNum,
          cwe: 'CWE-95',
          recommendation: 'Avoid eval() - use JSON.parse() for JSON or other safe alternatives',
        });
      }
      
      // Hardcoded secrets
      if (SECURITY_PATTERNS.hardcodedSecrets.test(line) && !line.includes('process.env')) {
        issues.push({
          severity: 'critical',
          type: 'Hardcoded Secret',
          message: 'Potential hardcoded secret or credential detected',
          filePath,
          line: lineNum,
          cwe: 'CWE-798',
          recommendation: 'Use environment variables or a secrets manager',
        });
      }
      
      // Command injection
      if (SECURITY_PATTERNS.commandInjection.test(line)) {
        issues.push({
          severity: 'high',
          type: 'Command Injection',
          message: 'Potential command injection via shell execution',
          filePath,
          line: lineNum,
          cwe: 'CWE-78',
          recommendation: 'Validate and sanitize all input to shell commands',
        });
      }
      
      // Path traversal
      const pathMatch = line.match(/['"`][^'"`]*\.\.\//);
      if (pathMatch && !line.includes('import') && !line.includes('require')) {
        issues.push({
          severity: 'medium',
          type: 'Path Traversal',
          message: 'Potential path traversal pattern detected',
          filePath,
          line: lineNum,
          cwe: 'CWE-22',
          recommendation: 'Validate and normalize file paths',
        });
      }
      
      // Prototype pollution
      if (SECURITY_PATTERNS.prototypePolluton.test(line)) {
        issues.push({
          severity: 'high',
          type: 'Prototype Pollution',
          message: 'Potential prototype pollution vulnerability',
          filePath,
          line: lineNum,
          cwe: 'CWE-1321',
          recommendation: 'Use Object.create(null) or validate object keys',
        });
      }
    });
    
    return issues;
  }

  /**
   * Detect code smells
   */
  private detectCodeSmells(sourceFile: SourceFile, filePath: string, content: string): CodeSmell[] {
    const smells: CodeSmell[] = [];
    const lines = content.split('\n');
    
    // Long file
    if (lines.length > CODE_SMELL_PATTERNS.longFile.threshold) {
      smells.push({
        type: 'Long File',
        severity: 'minor',
        message: `File has ${lines.length} lines (threshold: ${CODE_SMELL_PATTERNS.longFile.threshold})`,
        filePath,
        line: 1,
        suggestion: 'Consider splitting into multiple smaller files',
      });
    }
    
    // Check functions and methods
    sourceFile.forEachDescendant(node => {
      if (Node.isFunctionDeclaration(node) || Node.isMethodDeclaration(node) || Node.isArrowFunction(node)) {
        const funcLines = node.getEndLineNumber() - node.getStartLineNumber() + 1;
        const name = Node.isFunctionDeclaration(node) ? node.getName() : 
                     Node.isMethodDeclaration(node) ? node.getName() : 'arrow function';
        
        // Long function
        if (funcLines > CODE_SMELL_PATTERNS.longFunction.threshold) {
          smells.push({
            type: 'Long Function',
            severity: 'major',
            message: `Function "${name}" has ${funcLines} lines (threshold: ${CODE_SMELL_PATTERNS.longFunction.threshold})`,
            filePath,
            line: node.getStartLineNumber(),
            suggestion: 'Extract smaller helper functions',
          });
        }
        
        // Too many parameters
        const params = 'getParameters' in node ? node.getParameters() : [];
        if (params.length > CODE_SMELL_PATTERNS.tooManyParameters.threshold) {
          smells.push({
            type: 'Too Many Parameters',
            severity: 'minor',
            message: `Function "${name}" has ${params.length} parameters (threshold: ${CODE_SMELL_PATTERNS.tooManyParameters.threshold})`,
            filePath,
            line: node.getStartLineNumber(),
            suggestion: 'Consider using an options object or breaking down the function',
          });
        }
        
        // Deep nesting
        const maxDepth = this.getMaxNestingDepth(node);
        if (maxDepth > CODE_SMELL_PATTERNS.deepNesting.threshold) {
          smells.push({
            type: 'Deep Nesting',
            severity: 'major',
            message: `Function "${name}" has nesting depth of ${maxDepth} (threshold: ${CODE_SMELL_PATTERNS.deepNesting.threshold})`,
            filePath,
            line: node.getStartLineNumber(),
            suggestion: 'Use early returns, extract functions, or flatten conditionals',
          });
        }
      }
    });
    
    // Magic numbers
    lines.forEach((line, index) => {
      if (CODE_SMELL_PATTERNS.magicNumbers.pattern.test(line) && 
          !line.includes('const') && !line.includes('let') && !line.includes('=')) {
        smells.push({
          type: 'Magic Number',
          severity: 'info',
          message: 'Magic number detected',
          filePath,
          line: index + 1,
          suggestion: 'Extract to a named constant',
        });
      }
    });
    
    // TODO/FIXME comments
    lines.forEach((line, index) => {
      if (CODE_SMELL_PATTERNS.todoComments.pattern.test(line)) {
        smells.push({
          type: 'Unresolved TODO',
          severity: 'info',
          message: line.trim(),
          filePath,
          line: index + 1,
          suggestion: 'Address the TODO or create a tracking issue',
        });
      }
    });
    
    return smells;
  }

  /**
   * Detect refactoring opportunities
   */
  private detectRefactoringOpportunities(fileResults: Map<string, CodeAnalysisResult>): RefactoringOpportunity[] {
    const opportunities: RefactoringOpportunity[] = [];
    
    fileResults.forEach((result, filePath) => {
      // Long functions -> Extract function
      result.codeSmells
        .filter(s => s.type === 'Long Function')
        .forEach(smell => {
          opportunities.push({
            type: 'extract-function',
            description: smell.message,
            filePath,
            startLine: smell.line,
            endLine: smell.line + 50, // Estimate
            impact: 'medium',
          });
        });
      
      // Deep nesting -> Simplify conditional
      result.codeSmells
        .filter(s => s.type === 'Deep Nesting')
        .forEach(smell => {
          opportunities.push({
            type: 'simplify-conditional',
            description: smell.message,
            filePath,
            startLine: smell.line,
            endLine: smell.line + 20,
            impact: 'high',
          });
        });
      
      // Too many parameters -> Extract to object
      result.codeSmells
        .filter(s => s.type === 'Too Many Parameters')
        .forEach(smell => {
          opportunities.push({
            type: 'extract-variable',
            description: `${smell.message} - consider using options object`,
            filePath,
            startLine: smell.line,
            endLine: smell.line,
            impact: 'low',
          });
        });
    });
    
    return opportunities;
  }

  /**
   * Get function context for AI-assisted coding
   */
  async getFunctionContext(filePath: string, content: string, functionName: string): Promise<FunctionContext | null> {
    const analysis = await this.analyzeFile(filePath, content);
    const entity = analysis.entities.find(e => e.name === functionName);
    
    if (!entity) return null;
    
    // Extract what the function does based on its implementation
    const sourceFile = this.project.getSourceFile(filePath);
    if (!sourceFile) return null;
    
    let purpose = '';
    const funcNode = sourceFile.getFunction(functionName);
    if (funcNode) {
      const body = funcNode.getBody()?.getText() || '';
      purpose = this.inferFunctionPurpose(functionName, body, entity.parameters || []);
    }
    
    // Detect side effects
    const sideEffects = this.detectSideEffects(content, entity.startLine, entity.endLine);
    
    return {
      name: functionName,
      parameters: entity.parameters || [],
      returnType: entity.returnType || 'void',
      dependencies: entity.dependencies,
      sideEffects,
      complexity: entity.complexity || 1,
      purpose,
    };
  }

  /**
   * Semantic code search - find similar code patterns
   */
  async findSimilarCode(query: string, files: Array<{ path: string; content: string }>): Promise<Array<{
    filePath: string;
    startLine: number;
    endLine: number;
    code: string;
    similarity: number;
  }>> {
    const results: Array<{
      filePath: string;
      startLine: number;
      endLine: number;
      code: string;
      similarity: number;
    }> = [];
    
    const queryTokens = this.tokenize(query);
    
    for (const file of files) {
      const analysis = await this.analyzeFile(file.path, file.content);
      
      for (const entity of analysis.entities) {
        const entityText = `${entity.name} ${entity.signature || ''} ${entity.docComment || ''}`;
        const entityTokens = this.tokenize(entityText);
        const similarity = this.calculateTokenSimilarity(queryTokens, entityTokens);
        
        if (similarity > 0.3) {
          const lines = file.content.split('\n');
          const code = lines.slice(entity.startLine - 1, entity.endLine).join('\n');
          
          results.push({
            filePath: file.path,
            startLine: entity.startLine,
            endLine: entity.endLine,
            code,
            similarity,
          });
        }
      }
    }
    
    return results.sort((a, b) => b.similarity - a.similarity).slice(0, 10);
  }

  /**
   * Generate code documentation
   */
  async generateDocumentation(filePath: string, content: string): Promise<string> {
    const analysis = await this.analyzeFile(filePath, content);
    
    let doc = `# ${filePath.split('/').pop()}\n\n`;
    doc += `## Overview\n`;
    doc += `- Lines of code: ${analysis.complexity.linesOfCode}\n`;
    doc += `- Functions: ${analysis.complexity.functionCount}\n`;
    doc += `- Classes: ${analysis.complexity.classCount}\n`;
    doc += `- Complexity: ${analysis.complexity.cyclomaticComplexity}\n`;
    doc += `- Maintainability: ${analysis.complexity.maintainabilityIndex}%\n\n`;
    
    if (analysis.exports.length > 0) {
      doc += `## Exports\n`;
      for (const exp of analysis.exports) {
        doc += `- \`${exp.name}\` (${exp.type})\n`;
      }
      doc += '\n';
    }
    
    const functions = analysis.entities.filter(e => e.type === 'function' || e.type === 'arrow-function');
    if (functions.length > 0) {
      doc += `## Functions\n\n`;
      for (const func of functions) {
        doc += `### ${func.name}\n`;
        if (func.signature) doc += `\`\`\`typescript\n${func.signature}\n\`\`\`\n`;
        if (func.docComment) doc += `${func.docComment}\n`;
        if (func.parameters && func.parameters.length > 0) {
          doc += `**Parameters:**\n`;
          for (const param of func.parameters) {
            doc += `- \`${param.name}\`: ${param.type}${param.isOptional ? ' (optional)' : ''}\n`;
          }
        }
        if (func.returnType) doc += `**Returns:** ${func.returnType}\n`;
        doc += '\n';
      }
    }
    
    const classes = analysis.entities.filter(e => e.type === 'class');
    if (classes.length > 0) {
      doc += `## Classes\n\n`;
      for (const cls of classes) {
        doc += `### ${cls.name}\n`;
        if (cls.docComment) doc += `${cls.docComment}\n`;
        
        const methods = analysis.entities.filter(e => e.type === 'method' && e.name.startsWith(cls.name + '.'));
        if (methods.length > 0) {
          doc += `**Methods:**\n`;
          for (const method of methods) {
            doc += `- \`${method.name.split('.')[1]}\``;
            if (method.signature) doc += `: ${method.signature}`;
            doc += '\n';
          }
        }
        doc += '\n';
      }
    }
    
    return doc;
  }

  // Helper methods
  
  private getNestingDepth(node: Node): number {
    let depth = 0;
    let parent = node.getParent();
    while (parent) {
      if (Node.isIfStatement(parent) || Node.isForStatement(parent) || 
          Node.isWhileStatement(parent) || Node.isDoStatement(parent) ||
          Node.isTryStatement(parent)) {
        depth++;
      }
      parent = parent.getParent();
    }
    return depth;
  }

  private getMaxNestingDepth(node: Node): number {
    let maxDepth = 0;
    node.forEachDescendant(child => {
      const depth = this.getNestingDepth(child);
      maxDepth = Math.max(maxDepth, depth);
    });
    return maxDepth;
  }

  private getFunctionSignature(func: FunctionDeclaration): string {
    const params = func.getParameters().map(p => `${p.getName()}: ${p.getType().getText()}`).join(', ');
    const returnType = func.getReturnType().getText();
    const async = func.isAsync() ? 'async ' : '';
    return `${async}function ${func.getName()}(${params}): ${returnType}`;
  }

  private getClassSignature(cls: ClassDeclaration): string {
    const ext = cls.getExtends();
    const impl = cls.getImplements();
    let sig = `class ${cls.getName()}`;
    if (ext) sig += ` extends ${ext.getText()}`;
    if (impl.length > 0) sig += ` implements ${impl.map(i => i.getText()).join(', ')}`;
    return sig;
  }

  private getMethodSignature(method: MethodDeclaration): string {
    const params = method.getParameters().map(p => `${p.getName()}: ${p.getType().getText()}`).join(', ');
    const returnType = method.getReturnType().getText();
    const modifiers = method.getModifiers().map(m => m.getText()).join(' ');
    const async = method.isAsync() ? 'async ' : '';
    return `${modifiers} ${async}${method.getName()}(${params}): ${returnType}`.trim();
  }

  private getArrowSignature(arrow: ArrowFunction): string {
    const params = arrow.getParameters().map(p => `${p.getName()}: ${p.getType().getText()}`).join(', ');
    const returnType = arrow.getReturnType().getText();
    const async = arrow.isAsync() ? 'async ' : '';
    return `${async}(${params}) => ${returnType}`;
  }

  private getInterfaceSignature(iface: InterfaceDeclaration): string {
    const ext = iface.getExtends();
    let sig = `interface ${iface.getName()}`;
    if (ext.length > 0) sig += ` extends ${ext.map(e => e.getText()).join(', ')}`;
    return sig;
  }

  private calculateFunctionComplexity(func: FunctionDeclaration): number {
    let complexity = 1;
    func.forEachDescendant(node => {
      if ([SyntaxKind.IfStatement, SyntaxKind.ConditionalExpression, SyntaxKind.CaseClause,
           SyntaxKind.ForStatement, SyntaxKind.ForInStatement, SyntaxKind.ForOfStatement,
           SyntaxKind.WhileStatement, SyntaxKind.DoStatement, SyntaxKind.CatchClause].includes(node.getKind())) {
        complexity++;
      }
    });
    return complexity;
  }

  private calculateClassComplexity(cls: ClassDeclaration): number {
    let complexity = 0;
    cls.getMethods().forEach(method => {
      complexity += this.calculateMethodComplexity(method);
    });
    return complexity;
  }

  private calculateMethodComplexity(method: MethodDeclaration): number {
    let complexity = 1;
    method.forEachDescendant(node => {
      if ([SyntaxKind.IfStatement, SyntaxKind.ConditionalExpression, SyntaxKind.CaseClause,
           SyntaxKind.ForStatement, SyntaxKind.ForInStatement, SyntaxKind.ForOfStatement,
           SyntaxKind.WhileStatement, SyntaxKind.DoStatement, SyntaxKind.CatchClause].includes(node.getKind())) {
        complexity++;
      }
    });
    return complexity;
  }

  private calculateArrowComplexity(arrow: ArrowFunction): number {
    let complexity = 1;
    arrow.forEachDescendant(node => {
      if ([SyntaxKind.IfStatement, SyntaxKind.ConditionalExpression, SyntaxKind.CaseClause,
           SyntaxKind.ForStatement, SyntaxKind.ForInStatement, SyntaxKind.ForOfStatement,
           SyntaxKind.WhileStatement, SyntaxKind.DoStatement, SyntaxKind.CatchClause].includes(node.getKind())) {
        complexity++;
      }
    });
    return complexity;
  }

  private extractDependencies(node: Node): string[] {
    const deps: string[] = [];
    node.forEachDescendant(child => {
      if (Node.isIdentifier(child)) {
        const def = child.getDefinitions()[0];
        if (def) {
          const filePath = def.getSourceFile().getFilePath();
          if (!deps.includes(filePath)) deps.push(filePath);
        }
      }
    });
    return deps.filter(d => d !== node.getSourceFile().getFilePath());
  }

  private extractCalls(node: Node): string[] {
    const calls: string[] = [];
    node.forEachDescendant(child => {
      if (Node.isCallExpression(child)) {
        const expr = child.getExpression();
        calls.push(expr.getText());
      }
    });
    return [...new Set(calls)];
  }

  private extractClassDependencies(cls: ClassDeclaration): string[] {
    const deps: string[] = [];
    const ext = cls.getExtends();
    if (ext) deps.push(ext.getText());
    cls.getImplements().forEach(impl => deps.push(impl.getText()));
    return deps;
  }

  private extractArrowDependencies(arrow: ArrowFunction): string[] {
    return this.extractDependencies(arrow);
  }

  private extractArrowCalls(arrow: ArrowFunction): string[] {
    return this.extractCalls(arrow);
  }

  private extractInterfaceDependencies(iface: InterfaceDeclaration): string[] {
    return iface.getExtends().map(e => e.getText());
  }

  private updateDependencyGraph(filePath: string, imports: ImportInfo[], exports: ExportInfo[], entities: CodeEntity[]): void {
    const outgoingEdges = imports.map(i => i.moduleSpecifier).filter(m => m.startsWith('.') || m.startsWith('/'));
    
    this.dependencyGraph.nodes.set(filePath, {
      filePath,
      imports,
      exports,
      entities,
      incomingEdges: [],
      outgoingEdges,
    });
    
    // Update incoming edges for other nodes
    for (const edge of outgoingEdges) {
      const targetNode = this.dependencyGraph.nodes.get(edge);
      if (targetNode && !targetNode.incomingEdges.includes(filePath)) {
        targetNode.incomingEdges.push(filePath);
      }
    }
  }

  private buildCallGraph(sourceFile: SourceFile, filePath: string): void {
    sourceFile.forEachDescendant(node => {
      if (Node.isFunctionDeclaration(node) || Node.isMethodDeclaration(node)) {
        const name = node.getName() || 'anonymous';
        const fullName = `${filePath}:${name}`;
        
        const calls: { name: string; filePath: string; line: number }[] = [];
        node.forEachDescendant(child => {
          if (Node.isCallExpression(child)) {
            const expr = child.getExpression();
            calls.push({
              name: expr.getText(),
              filePath,
              line: child.getStartLineNumber(),
            });
          }
        });
        
        this.callGraph.set(fullName, {
          name,
          filePath,
          calls,
          calledBy: [],
        });
      }
    });
  }

  private resolveCrossFileReferences(): void {
    // Link calledBy relationships
    this.callGraph.forEach((node, key) => {
      node.calls.forEach(call => {
        this.callGraph.forEach((targetNode, targetKey) => {
          if (call.name === targetNode.name || call.name.endsWith(`.${targetNode.name}`)) {
            if (!targetNode.calledBy.find(c => c.name === node.name && c.filePath === node.filePath)) {
              targetNode.calledBy.push({
                name: node.name,
                filePath: node.filePath,
                line: call.line,
              });
            }
          }
        });
      });
    });
  }

  private generateSummary(entities: CodeEntity[], complexity: ComplexityMetrics, issues: CodeIssue[], securityIssues: SecurityIssue[], codeSmells: CodeSmell[]): AnalysisSummary {
    const errorCount = issues.filter(i => i.type === 'error').length;
    const warningCount = issues.filter(i => i.type === 'warning').length;
    const criticalSecurity = securityIssues.filter(s => s.severity === 'critical' || s.severity === 'high').length;
    const majorSmells = codeSmells.filter(s => s.severity === 'major').length;
    
    // Calculate health score (0-100)
    let healthScore = 100;
    healthScore -= errorCount * 10;
    healthScore -= warningCount * 2;
    healthScore -= criticalSecurity * 15;
    healthScore -= majorSmells * 5;
    healthScore -= Math.max(0, complexity.cyclomaticComplexity - 20);
    healthScore = Math.max(0, Math.min(100, healthScore));
    
    const topIssues: string[] = [];
    if (errorCount > 0) topIssues.push(`${errorCount} type errors`);
    if (criticalSecurity > 0) topIssues.push(`${criticalSecurity} critical security issues`);
    if (majorSmells > 0) topIssues.push(`${majorSmells} major code smells`);
    
    return {
      totalFiles: 1,
      totalLines: complexity.linesOfCode,
      totalFunctions: complexity.functionCount,
      totalClasses: complexity.classCount,
      averageComplexity: complexity.cyclomaticComplexity,
      healthScore,
      topIssues,
    };
  }

  private generateProjectSummary(fileResults: Map<string, CodeAnalysisResult>): AnalysisSummary {
    let totalLines = 0;
    let totalFunctions = 0;
    let totalClasses = 0;
    let totalComplexity = 0;
    let totalHealthScore = 0;
    const allTopIssues: string[] = [];
    
    fileResults.forEach(result => {
      totalLines += result.complexity.linesOfCode;
      totalFunctions += result.complexity.functionCount;
      totalClasses += result.complexity.classCount;
      totalComplexity += result.complexity.cyclomaticComplexity;
      totalHealthScore += result.summary.healthScore;
      allTopIssues.push(...result.summary.topIssues);
    });
    
    const fileCount = fileResults.size;
    
    return {
      totalFiles: fileCount,
      totalLines,
      totalFunctions,
      totalClasses,
      averageComplexity: fileCount > 0 ? Math.round(totalComplexity / fileCount) : 0,
      healthScore: fileCount > 0 ? Math.round(totalHealthScore / fileCount) : 100,
      topIssues: [...new Set(allTopIssues)].slice(0, 5),
    };
  }

  private inferFunctionPurpose(name: string, body: string, params: ParameterInfo[]): string {
    // Infer purpose from function name patterns
    const lowerName = name.toLowerCase();
    
    if (lowerName.startsWith('get') || lowerName.startsWith('fetch')) {
      return `Retrieves ${name.slice(3).replace(/([A-Z])/g, ' $1').trim().toLowerCase()}`;
    }
    if (lowerName.startsWith('set') || lowerName.startsWith('update')) {
      return `Updates ${name.slice(3).replace(/([A-Z])/g, ' $1').trim().toLowerCase()}`;
    }
    if (lowerName.startsWith('is') || lowerName.startsWith('has') || lowerName.startsWith('can')) {
      return `Checks ${name.slice(2).replace(/([A-Z])/g, ' $1').trim().toLowerCase()}`;
    }
    if (lowerName.startsWith('create') || lowerName.startsWith('make') || lowerName.startsWith('build')) {
      return `Creates ${name.slice(6).replace(/([A-Z])/g, ' $1').trim().toLowerCase()}`;
    }
    if (lowerName.startsWith('handle') || lowerName.startsWith('on')) {
      return `Handles ${name.slice(6).replace(/([A-Z])/g, ' $1').trim().toLowerCase()} event`;
    }
    if (lowerName.includes('validate') || lowerName.includes('check')) {
      return `Validates input data`;
    }
    if (lowerName.includes('transform') || lowerName.includes('convert') || lowerName.includes('parse')) {
      return `Transforms or parses data`;
    }
    
    return `Function with ${params.length} parameters`;
  }

  private detectSideEffects(content: string, startLine: number, endLine: number): string[] {
    const lines = content.split('\n').slice(startLine - 1, endLine);
    const code = lines.join('\n');
    const sideEffects: string[] = [];
    
    if (/console\.(log|warn|error|info)/.test(code)) sideEffects.push('console output');
    if (/localStorage|sessionStorage/.test(code)) sideEffects.push('browser storage');
    if (/fetch\(|axios|XMLHttpRequest/.test(code)) sideEffects.push('network request');
    if (/document\.|window\./.test(code)) sideEffects.push('DOM manipulation');
    if (/setState|dispatch|emit/.test(code)) sideEffects.push('state mutation');
    if (/fs\.|writeFile|readFile/.test(code)) sideEffects.push('file system');
    
    return sideEffects;
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 2);
  }

  private calculateTokenSimilarity(a: string[], b: string[]): number {
    const setA = new Set(a);
    const setB = new Set(b);
    const intersection = [...setA].filter(x => setB.has(x)).length;
    const union = new Set([...a, ...b]).size;
    return union > 0 ? intersection / union : 0;
  }

  /**
   * Get dependency graph for visualization
   */
  getDependencyGraph(): DependencyGraph {
    return this.dependencyGraph;
  }

  /**
   * Get call graph for tracing
   */
  getCallGraph(): Map<string, CallGraphNode> {
    return this.callGraph;
  }

  /**
   * Clear cached analysis
   */
  clearCache(): void {
    this.entityCache.clear();
    this.dependencyGraph = { nodes: new Map(), edges: [] };
    this.callGraph.clear();
    this.project = new Project({
      useInMemoryFileSystem: true,
      compilerOptions: {
        target: 99,
        module: 99,
        moduleResolution: 2,
        strict: true,
        esModuleInterop: true,
        skipLibCheck: true,
        allowJs: true,
        jsx: 4,
      },
    });
  }
}

// Singleton instance
export const codeAnalyzer = new CodeAnalyzer();
