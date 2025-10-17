/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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

#ifndef re2c_ins_h
#define re2c_ins_h

#include "tools/re2c/basics.h"

#define nChars 256
typedef unsigned char Char;

#define CHAR 0
#define GOTO 1
#define FORK 2
#define TERM 3
#define CTXT 4

typedef union Ins {
    struct {
	byte	tag;
	byte	marked;
	void	*link;
    }			i;
    struct {
	unsigned short	value;
	unsigned short	bump;
	void	*link;
    }			c;
} Ins;

static int isMarked(Ins *i){
    return i->i.marked != 0;
}

static void mark(Ins *i){
    i->i.marked = 1;
}

static void unmark(Ins *i){
    i->i.marked = 0;
}

#endif
