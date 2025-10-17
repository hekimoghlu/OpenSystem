/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: head/bin/sh/arith_yylex.c 279503 2015-03-01 21:46:55Z jilles $");

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "shell.h"
#include "arith_yacc.h"
#include "expand.h"
#include "error.h"
#include "memalloc.h"
#include "parser.h"
#include "syntax.h"

#if ARITH_BOR + 11 != ARITH_BORASS || ARITH_ASS + 11 != ARITH_EQ
#error Arithmetic tokens are out of order.
#endif

int
yylex(void)
{
	int value;
	const char *buf = arith_buf;
	char *end;
	const char *p;

	for (;;) {
		value = *buf;
		switch (value) {
		case ' ':
		case '\t':
		case '\n':
			buf++;
			continue;
		default:
			return ARITH_BAD;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			yylval.val = strtoarith_t(buf, &end, 0);
			arith_buf = end;
			return ARITH_NUM;
		case 'A':
		case 'B':
		case 'C':
		case 'D':
		case 'E':
		case 'F':
		case 'G':
		case 'H':
		case 'I':
		case 'J':
		case 'K':
		case 'L':
		case 'M':
		case 'N':
		case 'O':
		case 'P':
		case 'Q':
		case 'R':
		case 'S':
		case 'T':
		case 'U':
		case 'V':
		case 'W':
		case 'X':
		case 'Y':
		case 'Z':
		case '_':
		case 'a':
		case 'b':
		case 'c':
		case 'd':
		case 'e':
		case 'f':
		case 'g':
		case 'h':
		case 'i':
		case 'j':
		case 'k':
		case 'l':
		case 'm':
		case 'n':
		case 'o':
		case 'p':
		case 'q':
		case 'r':
		case 's':
		case 't':
		case 'u':
		case 'v':
		case 'w':
		case 'x':
		case 'y':
		case 'z':
			p = buf;
			while (buf++, is_in_name(*buf))
				;
			yylval.name = stalloc(buf - p + 1);
			memcpy(yylval.name, p, buf - p);
			yylval.name[buf - p] = '\0';
			value = ARITH_VAR;
			goto out;
		case '=':
			value += ARITH_ASS - '=';
checkeq:
			buf++;
checkeqcur:
			if (*buf != '=')
				goto out;
			value += 11;
			break;
		case '>':
			switch (*++buf) {
			case '=':
				value += ARITH_GE - '>';
				break;
			case '>':
				value += ARITH_RSHIFT - '>';
				goto checkeq;
			default:
				value += ARITH_GT - '>';
				goto out;
			}
			break;
		case '<':
			switch (*++buf) {
			case '=':
				value += ARITH_LE - '<';
				break;
			case '<':
				value += ARITH_LSHIFT - '<';
				goto checkeq;
			default:
				value += ARITH_LT - '<';
				goto out;
			}
			break;
		case '|':
			if (*++buf != '|') {
				value += ARITH_BOR - '|';
				goto checkeqcur;
			}
			value += ARITH_OR - '|';
			break;
		case '&':
			if (*++buf != '&') {
				value += ARITH_BAND - '&';
				goto checkeqcur;
			}
			value += ARITH_AND - '&';
			break;
		case '!':
			if (*++buf != '=') {
				value += ARITH_NOT - '!';
				goto out;
			}
			value += ARITH_NE - '!';
			break;
		case 0:
			goto out;
		case '(':
			value += ARITH_LPAREN - '(';
			break;
		case ')':
			value += ARITH_RPAREN - ')';
			break;
		case '*':
			value += ARITH_MUL - '*';
			goto checkeq;
		case '/':
			value += ARITH_DIV - '/';
			goto checkeq;
		case '%':
			value += ARITH_REM - '%';
			goto checkeq;
		case '+':
			if (buf[1] == '+')
				return ARITH_BAD;
			value += ARITH_ADD - '+';
			goto checkeq;
		case '-':
			if (buf[1] == '-')
				return ARITH_BAD;
			value += ARITH_SUB - '-';
			goto checkeq;
		case '~':
			value += ARITH_BNOT - '~';
			break;
		case '^':
			value += ARITH_BXOR - '^';
			goto checkeq;
		case '?':
			value += ARITH_QMARK - '?';
			break;
		case ':':
			value += ARITH_COLON - ':';
			break;
		}
		break;
	}

	buf++;
out:
	arith_buf = buf;
	return value;
}
