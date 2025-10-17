/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
 * Glenn Fowler
 * AT&T Research
 *
 * magic interface definitions
 */

#ifndef _MAGIC_H
#define _MAGIC_H

#include <sfio.h>
#include <ls.h>

#define MAGIC_VERSION	19961031L

#ifndef MAGIC_FILE
#define MAGIC_FILE	"lib/file/magic"
#endif

#ifndef MAGIC_DIR
#define MAGIC_DIR	"lib/file"
#endif

#define MAGIC_FILE_ENV	"MAGICFILE"

#define MAGIC_MIME	(1<<0)		/* magictype returns MIME type	*/
#define MAGIC_VERBOSE	(1<<1)		/* verbose magic file errors	*/
#define MAGIC_ALL	(1<<2)		/* list all table matches	*/

#define MAGIC_USER	(1L<<16)	/* first user flag bit		*/

struct Magic_s;
struct Magicdisc_s;

typedef struct Magicdisc_s
{
	unsigned long	version;	/* interface version		*/
	unsigned long	flags;		/* MAGIC_* flags		*/
	Error_f		errorf;		/* error function		*/
} Magicdisc_t;

typedef struct Magic_s
{
	const char*	id;		/* library id string		*/

#ifdef _MAGIC_PRIVATE_
	_MAGIC_PRIVATE_
#endif

} Magic_t;

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern Magic_t*		magicopen(Magicdisc_t*);
extern int		magicload(Magic_t*, const char*, unsigned long);
extern int		magiclist(Magic_t*, Sfio_t*);
extern char*		magictype(Magic_t*, Sfio_t*, const char*, struct stat*);
extern int		magicclose(Magic_t*);

#undef	extern

#endif
