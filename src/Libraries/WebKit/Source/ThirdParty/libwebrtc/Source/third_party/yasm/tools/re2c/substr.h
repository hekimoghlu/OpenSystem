/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

#ifndef re2c_substr_h
#define re2c_substr_h

#include <stdio.h>
#include <stdlib.h>
#include "tools/re2c/basics.h"

struct SubStr {
    char		*str;
    unsigned int	len;
};

typedef struct SubStr SubStr;

int SubStr_eq(const SubStr *, const SubStr *);

static void SubStr_init_u(SubStr*, unsigned char*, unsigned int);
static SubStr *SubStr_new_u(unsigned char*, unsigned int);

static void SubStr_init(SubStr*, char*, unsigned int);
static SubStr *SubStr_new(char*, unsigned int);

static void SubStr_copy(SubStr*, const SubStr*);
static SubStr *SubStr_new_copy(const SubStr*);

void SubStr_out(const SubStr*, FILE *);
#define SubStr_delete(x)    free(x)

typedef struct SubStr Str;

void Str_init(Str*, const SubStr*);
Str *Str_new(const SubStr*);

void Str_copy(Str*, Str*);
Str *Str_new_copy(Str*);

Str *Str_new_empty(void);
void Str_destroy(Str *);
void Str_delete(Str *);

static void
SubStr_init_u(SubStr *r, unsigned char *s, unsigned int l)
{
    r->str = (char*)s;
    r->len = l;
}

static SubStr *
SubStr_new_u(unsigned char *s, unsigned int l)
{
    SubStr *r = malloc(sizeof(SubStr));
    r->str = (char*)s;
    r->len = l;
    return r;
}

static void
SubStr_init(SubStr *r, char *s, unsigned int l)
{
    r->str = s;
    r->len = l;
}

static SubStr *
SubStr_new(char *s, unsigned int l)
{
    SubStr *r = malloc(sizeof(SubStr));
    r->str = s;
    r->len = l;
    return r;
}

static void
SubStr_copy(SubStr *r, const SubStr *s)
{
    r->str = s->str;
    r->len = s->len;
}

static SubStr *
SubStr_new_copy(const SubStr *s)
{
    SubStr *r = malloc(sizeof(SubStr));
    r->str = s->str;
    r->len = s->len;
    return r;
}

#endif
