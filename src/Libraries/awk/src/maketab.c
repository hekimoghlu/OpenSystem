/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/*
 * this program makes the table to link function names
 * and type indices that is used by execute() in run.c.
 * it finds the indices in awkgram.tab.h, produced by bison.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "awk.h"
#include "awkgram.tab.h"

struct xx
{	int token;
	const char *name;
	const char *pname;
} proc[] = {
	{ PROGRAM, "program", NULL },
	{ BOR, "boolop", " || " },
	{ AND, "boolop", " && " },
	{ NOT, "boolop", " !" },
	{ NE, "relop", " != " },
	{ EQ, "relop", " == " },
	{ LE, "relop", " <= " },
	{ LT, "relop", " < " },
	{ GE, "relop", " >= " },
	{ GT, "relop", " > " },
	{ ARRAY, "array", NULL },
	{ INDIRECT, "indirect", "$(" },
	{ SUBSTR, "substr", "substr" },
	{ SUB, "sub", "sub" },
	{ GSUB, "gsub", "gsub" },
	{ INDEX, "sindex", "sindex" },
	{ SPRINTF, "awksprintf", "sprintf " },
	{ ADD, "arith", " + " },
	{ MINUS, "arith", " - " },
	{ MULT, "arith", " * " },
	{ DIVIDE, "arith", " / " },
	{ MOD, "arith", " % " },
	{ UMINUS, "arith", " -" },
	{ UPLUS, "arith", " +" },
	{ POWER, "arith", " **" },
	{ PREINCR, "incrdecr", "++" },
	{ POSTINCR, "incrdecr", "++" },
	{ PREDECR, "incrdecr", "--" },
	{ POSTDECR, "incrdecr", "--" },
	{ CAT, "cat", " " },
	{ PASTAT, "pastat", NULL },
	{ PASTAT2, "dopa2", NULL },
	{ MATCH, "matchop", " ~ " },
	{ NOTMATCH, "matchop", " !~ " },
	{ MATCHFCN, "matchop", "matchop" },
	{ INTEST, "intest", "intest" },
	{ PRINTF, "awkprintf", "printf" },
	{ PRINT, "printstat", "print" },
	{ CLOSE, "closefile", "closefile" },
	{ DELETE, "awkdelete", "awkdelete" },
	{ SPLIT, "split", "split" },
	{ ASSIGN, "assign", " = " },
	{ ADDEQ, "assign", " += " },
	{ SUBEQ, "assign", " -= " },
	{ MULTEQ, "assign", " *= " },
	{ DIVEQ, "assign", " /= " },
	{ MODEQ, "assign", " %= " },
	{ POWEQ, "assign", " ^= " },
	{ CONDEXPR, "condexpr", " ?: " },
	{ IF, "ifstat", "if(" },
	{ WHILE, "whilestat", "while(" },
	{ FOR, "forstat", "for(" },
	{ DO, "dostat", "do" },
	{ IN, "instat", "instat" },
	{ NEXT, "jump", "next" },
	{ NEXTFILE, "jump", "nextfile" },
	{ EXIT, "jump", "exit" },
	{ BREAK, "jump", "break" },
	{ CONTINUE, "jump", "continue" },
	{ RETURN, "jump", "ret" },
	{ BLTIN, "bltin", "bltin" },
	{ CALL, "call", "call" },
	{ ARG, "arg", "arg" },
	{ VARNF, "getnf", "NF" },
	{ GETLINE, "awkgetline", "getline" },
	{ 0, "", "" },
};

#define SIZE	(LASTTOKEN - FIRSTTOKEN + 1)
const char *table[SIZE];
char *names[SIZE];

int main(int argc, char *argv[])
{
	const struct xx *p;
	int i, n, tok;
	char c;
	FILE *fp;
	char buf[200], name[200], def[200];
	enum { TOK_UNKNOWN, TOK_ENUM, TOK_DEFINE } tokentype = TOK_UNKNOWN;

	printf("#include <stdio.h>\n");
	printf("#include \"awk.h\"\n");
	printf("#include \"awkgram.tab.h\"\n\n");

	if (argc != 2) {
		fprintf(stderr, "usage: maketab YTAB_H\n");
		exit(1);
	}
	if ((fp = fopen(argv[1], "r")) == NULL) {
		fprintf(stderr, "maketab can't open %s!\n", argv[1]);
		exit(1);
	}
	printf("static const char * const printname[%d] = {\n", SIZE);
	i = 0;
	while (fgets(buf, sizeof buf, fp) != NULL) {
		// 199 is sizeof(def) - 1
		if (tokentype != TOK_ENUM) {
			n = sscanf(buf, "%1c %199s %199s %d", &c, def, name,
			    &tok);
			if (n == 4 && c == '#' && strcmp(def, "define") == 0) {
				tokentype = TOK_DEFINE;
			} else if (tokentype != TOK_UNKNOWN) {
				continue;
			}
		}
		if (tokentype != TOK_DEFINE) {
			/* not a valid #define, bison uses enums now */
			n = sscanf(buf, "%199s = %d,\n", name, &tok);
			if (n != 2)
				continue;
			tokentype = TOK_ENUM;
		}
		if (strcmp(name, "YYSTYPE_IS_DECLARED") == 0) {
			tokentype = TOK_UNKNOWN;
			continue;
		}
		if (tok < FIRSTTOKEN || tok > LASTTOKEN) {
			tokentype = TOK_UNKNOWN;
			/* fprintf(stderr, "maketab funny token %d %s ignored\n", tok, buf); */
			continue;
		}
		names[tok-FIRSTTOKEN] = strdup(name);
		if (names[tok-FIRSTTOKEN] == NULL) {
			fprintf(stderr, "maketab out of space copying %s", name);
			continue;
		}
		printf("\t\"%s\",\t/* %d */\n", name, tok);
		i++;
	}
	printf("};\n\n");

	for (p=proc; p->token!=0; p++)
		table[p->token-FIRSTTOKEN] = p->name;
	printf("\nCell *(*proctab[%d])(Node **, int) = {\n", SIZE);
	for (i=0; i<SIZE; i++)
		printf("\t%s,\t/* %s */\n",
		    table[i] ? table[i] : "nullproc", names[i] ? names[i] : "");
	printf("};\n\n");

	printf("const char *tokname(int n)\n");	/* print a tokname() function */
	printf("{\n");
	printf("\tstatic char buf[100];\n\n");
	printf("\tif (n < FIRSTTOKEN || n > LASTTOKEN) {\n");
	printf("\t\tsnprintf(buf, sizeof(buf), \"token %%d\", n);\n");
	printf("\t\treturn buf;\n");
	printf("\t}\n");
	printf("\treturn printname[n-FIRSTTOKEN];\n");
	printf("}\n");
	return 0;
}
