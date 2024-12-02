import antlr4
from project.GraphQueryLexer import GraphQueryLexer
from project.GraphQueryParser import GraphQueryParser


def program_to_tree(program: str) -> tuple[antlr4.ParserRuleContext, bool]:
    sanitized_program = program.replace("<EOF>", "EOF")
    lexer = GraphQueryLexer(antlr4.InputStream(sanitized_program))
    parser = GraphQueryParser(antlr4.CommonTokenStream(lexer))
    parser.removeParseListeners()
    parse_tree = parser.prog()

    return (parse_tree, parser.getNumberOfSyntaxErrors() == 0)


def tree_to_program(tree: antlr4.ParserRuleContext) -> str:
    if not tree:
        return ""

    class TreeTokensCollector(antlr4.ParseTreeListener):
        def __init__(self):
            self.tokens = []

        def visitTerminal(self, node):
            self.tokens.append(node.getText())

    collector = TreeTokensCollector()
    antlr4.ParseTreeWalker().walk(collector, tree)

    return " ".join(collector.tokens)


def nodes_count(tree: antlr4.ParserRuleContext) -> int:
    if not tree:
        return 0

    class NodeCounter(antlr4.ParseTreeListener):
        def __init__(self):
            self.count = 0

        def enterEveryRule(self, ctx):
            self.count += 1

    counter = NodeCounter()
    antlr4.ParseTreeWalker().walk(counter, tree)

    return counter.count
