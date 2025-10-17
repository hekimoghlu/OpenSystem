/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#ifndef RE2C_PARSER_H
#define RE2C_PARSER_H

/* Tokens */
enum yytokentype {
    CLOSESIZE = 258,
    CLOSE = 259,
    ID = 260,
    CODE = 261,
    RANGE = 262,
    STRING = 263,
    NONE = 264
};

#define CLOSESIZE 258
#define CLOSE 259
#define ID 260
#define CODE 261
#define RANGE 262
#define STRING 263
#define NONE 264

typedef union {
    Symbol	*symbol;
    RegExp	*regexp;
    Token	*token;
    char	op;
    ExtOp	extop;
} yystype;

extern yystype yylval;

#endif
