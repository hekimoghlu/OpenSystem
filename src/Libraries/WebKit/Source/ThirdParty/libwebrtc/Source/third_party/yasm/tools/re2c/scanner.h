/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

#ifndef _scanner_h
#define	_scanner_h

#include <stdio.h>
#include "tools/re2c/token.h"

typedef struct Scanner {
    FILE		*in;
    unsigned char	*bot, *tok, *ptr, *cur, *pos, *lim, *top, *eof;
    unsigned int	tchar, tline, cline;
} Scanner;

void Scanner_init(Scanner*, FILE *);
static Scanner *Scanner_new(FILE *);

int Scanner_echo(Scanner*, FILE *);
int Scanner_scan(Scanner*);
void Scanner_fatal(Scanner*, const char*);
static SubStr Scanner_token(Scanner*);
static unsigned int Scanner_line(Scanner*);

static SubStr
Scanner_token(Scanner *s)
{
    SubStr r;
    SubStr_init_u(&r, s->tok, s->cur - s->tok);
    return r;
}

static unsigned int
Scanner_line(Scanner *s)
{
    return s->cline;
}

static Scanner *
Scanner_new(FILE *i)
{
    Scanner *r = malloc(sizeof(Scanner));
    Scanner_init(r, i);
    return r;
}

#endif
