grammar GraphQuery;

// Parser
prog : stmt* ;

stmt
    : declare
    | bind
    | add
    | remove
    ;

declare : LET varName IS GRAPH ;

bind : LET varName ASSIGN expr ;

remove : REMOVE (VERTEX | EDGE | VERTICES) expr FROM varName ;

add : ADD (VERTEX | EDGE) expr TO varName ;

expr
    : NUM                   # NumberExpr
    | CHAR                  # CharExpr
    | varName               # VarExpr
    | edge_expr             # EdgeExpr
    | set_expr              # SetExpr
    | regexp                # RegexpExpr
    | select                # SelectExpr
    ;

set_expr : LBRACKET expr (COMMA expr)* RBRACKET ;

edge_expr : LPAREN expr COMMA expr COMMA expr RPAREN ;

regexp
    : regexp PIPE regexp    # RegexpOr
    | regexp AND regexp     # RegexpAnd
    | regexp DOT regexp     # RegexpConcat
    | regexp CARET range    # RegexpRange
    | LPAREN regexp RPAREN  # RegexpParens
    | CHAR                  # RegexpChar
    | varName               # RegexpVar
    ;

range : LBRACKET NUM (DOTDOT NUM?)? RBRACKET ;

select
    : v_filter? v_filter? RETURN varName (COMMA varName)?
      WHERE varName REACHABLE FROM varName IN varName BY expr
    ;

v_filter : FOR varName IN expr ;

// Lexer
LET : 'let' ;
IS : 'is' ;
GRAPH : 'graph' ;
REMOVE : 'remove' ;
VERTEX : 'vertex' ;
EDGE : 'edge' ;
VERTICES : 'vertices' ;
ADD : 'add' ;
TO : 'to' ;
FROM : 'from' ;
FOR : 'for' ;
IN : 'in' ;
RETURN : 'return' ;
WHERE : 'where' ;
REACHABLE : 'reachable' ;
BY : 'by' ;

ASSIGN : '=' ;
LBRACKET : '[' ;
RBRACKET : ']' ;
LPAREN : '(' ;
RPAREN : ')' ;
COMMA : ',' ;
DOTDOT : '..' ;
PIPE : '|' ;
AND : '&' ;
DOT : '.' ;
CARET : '^' ;

varName : IDENTIFIER ;
IDENTIFIER : [a-zA-Z] [a-zA-Z0-9]* ;
NUM : '0' | [1-9][0-9]* ;
CHAR : '"' [a-z] '"' | '\'' [a-z] '\'' ;

WS : [ \t\r\n]+ -> skip ;