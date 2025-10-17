/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#pragma prototyped
/*
 * David Korn
 * AT&T Bell Laboratories
 *
 * header for wc library interface
 */

#ifndef _WC_H
#define _WC_H

#include <ast.h>

#define WC_LINES	0x01
#define WC_WORDS	0x02
#define WC_CHARS	0x04
#define WC_MBYTE	0x08
#define WC_LONGEST	0x10
#define WC_QUIET	0x20
#define WC_NOUTF8	0x40

typedef struct
{
	char	type[1<<CHAR_BIT];
	Sfoff_t words;
	Sfoff_t lines;
	Sfoff_t chars;
	Sfoff_t longest;
	int	mode;
	int	mb;
} Wc_t;

#define wc_count	_cmd_wccount
#define wc_init		_cmd_wcinit

extern Wc_t*		wc_init(int);
extern int		wc_count(Wc_t*, Sfio_t*, const char*);

#endif /* _WC_H */
