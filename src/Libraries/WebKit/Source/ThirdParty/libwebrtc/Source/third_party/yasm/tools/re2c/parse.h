/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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

#ifndef re2c_parse_h
#define re2c_parse_h

#include <stdio.h>
#include "tools/re2c/scanner.h"
#include "tools/re2c/re.h"

typedef struct Symbol {
    struct Symbol		*next;
    Str			name;
    RegExp		*re;
} Symbol;

void Symbol_init(Symbol *, const SubStr*);
static Symbol *Symbol_new(const SubStr*);
Symbol *Symbol_find(const SubStr*);

void line_source(FILE *, unsigned int);
void parse(FILE *, FILE *);

static Symbol *
Symbol_new(const SubStr *str)
{
    Symbol *r = malloc(sizeof(Symbol));
    Symbol_init(r, str);
    return r;
}

#endif
