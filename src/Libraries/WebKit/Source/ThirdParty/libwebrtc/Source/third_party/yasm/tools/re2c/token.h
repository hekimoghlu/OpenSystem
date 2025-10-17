/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#ifndef re2c_token_h
#define	re2c_token_h

#include "substr.h"

typedef struct Token {
    Str			text;
    unsigned int	line;
} Token;

static void Token_init(Token *, SubStr, unsigned int);
static Token *Token_new(SubStr, unsigned int);

static void
Token_init(Token *r, SubStr t, unsigned int l)
{
    Str_copy(&r->text, &t);
    r->line = l;
}

static Token *
Token_new(SubStr t, unsigned int l)
{
    Token *r = malloc(sizeof(Token));
    Str_init(&r->text, &t);
    r->line = l;
    return r;
}

#endif
