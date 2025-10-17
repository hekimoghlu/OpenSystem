/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
 * mime/mailcap internal interface
 */

#ifndef _MIMELIB_H
#define _MIMELIB_H	1

#include <ast.h>
#include <cdt.h>
#include <magic.h>
#include <tok.h>

struct Mime_s;

typedef void (*Free_f)(struct Mime_s*);

#define _MIME_PRIVATE_ \
	Mimedisc_t*	disc;		/* mime discipline		*/ \
	Dtdisc_t	dict;		/* cdt discipline		*/ \
	Magicdisc_t	magicd;		/* magic discipline		*/ \
	Dt_t*		cap;		/* capability tree		*/ \
	Sfio_t*		buf;		/* string buffer		*/ \
	Magic_t*	magic;		/* mimetype() magic handle	*/ \
	Free_f		freef;		/* avoid magic lib if possible	*/ \

#include <mime.h>
#include <ctype.h>

#endif
