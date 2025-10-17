/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include <sys/types.h>
#include <db.h>
#include <fcntl.h>

#define CHARMAP_SYMBOL_LEN 64
#define BUFSIZE 80

#define NOTEXISTS	0
#define EXISTS		1

#define	SYMBOL_CHAR	0
#define	SYMBOL_CHAIN	1
#define	SYMBOL_SYMBOL	2
#define	SYMBOL_STRING	3
#define	SYMBOL_IGNORE	4
#define	SYMBOL_ELLIPSIS	5
struct symbol {
	int type;
	int val;
	wchar_t name[CHARMAP_SYMBOL_LEN];
	union {
		wchar_t wc;
		wchar_t str[STR_LEN];
	} u;
};

extern int line_no;

struct symbol *getsymbol(const wchar_t *, int);
extern char *showwcs(const wchar_t *, int);

extern char map_name[FILENAME_MAX];
