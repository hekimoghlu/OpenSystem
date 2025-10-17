/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#define ARITH_ASS 1

#define ARITH_OR 2
#define ARITH_AND 3
#define ARITH_BAD 4
#define ARITH_NUM 5
#define ARITH_VAR 6
#define ARITH_NOT 7

#define ARITH_BINOP_MIN 8
#define ARITH_LE 8
#define ARITH_GE 9
#define ARITH_LT 10
#define ARITH_GT 11
#define ARITH_EQ 12
#define ARITH_REM 13
#define ARITH_BAND 14
#define ARITH_LSHIFT 15
#define ARITH_RSHIFT 16
#define ARITH_MUL 17
#define ARITH_ADD 18
#define ARITH_BOR 19
#define ARITH_SUB 20
#define ARITH_BXOR 21
#define ARITH_DIV 22
#define ARITH_NE 23
#define ARITH_BINOP_MAX 24

#define ARITH_ASS_MIN 24
#define ARITH_REMASS 24
#define ARITH_BANDASS 25
#define ARITH_LSHIFTASS 26
#define ARITH_RSHIFTASS 27
#define ARITH_MULASS 28
#define ARITH_ADDASS 29
#define ARITH_BORASS 30
#define ARITH_SUBASS 31
#define ARITH_BXORASS 32
#define ARITH_DIVASS 33
#define ARITH_ASS_MAX 34

#define ARITH_LPAREN 34
#define ARITH_RPAREN 35
#define ARITH_BNOT 36
#define ARITH_QMARK 37
#define ARITH_COLON 38

extern const char *arith_buf;

union yystype {
	arith_t val;
	char *name;
};

extern union yystype yylval;

int yylex(void);
