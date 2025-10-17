/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
 * mime/mailcap interface
 */

#ifndef _MIMETYPE_H
#define _MIMETYPE_H	1

#include <sfio.h>
#include <ls.h>

#define MIME_VERSION	19970717L

#ifndef MIME_FILES
#define MIME_FILES	"~/.mailcap:/usr/local/etc/mailcap:/usr/etc/mailcap:/etc/mailcap:/etc/mail/mailcap:/usr/public/lib/mailcap"
#endif

#define MIME_FILES_ENV	"MAILCAP"

#define MIME_LIST	(1<<0)		/* mimeload arg is : list	*/
#define MIME_NOMAGIC	(1<<1)		/* no magic for mimetype()	*/
#define MIME_PIPE	(1<<2)		/* mimeview() io is piped	*/
#define MIME_REPLACE	(1<<3)		/* replace existing definition	*/

#define MIME_USER	(1L<<16)	/* first user flag bit		*/

struct Mime_s;
typedef struct Mime_s Mime_t;

struct Mimedisc_s;
typedef struct Mimedisc_s Mimedisc_t;

typedef int (*Mimevalue_f)(Mime_t*, void*, char*, size_t, Mimedisc_t*);

struct Mimedisc_s
{
	unsigned long	version;	/* interface version		*/
	unsigned long	flags;		/* MIME_* flags			*/
	Error_f		errorf;		/* error function		*/
	Mimevalue_f	valuef;		/* value extraction function	*/
};

struct Mime_s
{
	const char*	id;		/* library id string		*/

#ifdef _MIME_PRIVATE_
	_MIME_PRIVATE_
#endif

};

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern Mime_t*	mimeopen(Mimedisc_t*);
extern int	mimeload(Mime_t*, const char*, unsigned long);
extern int	mimelist(Mime_t*, Sfio_t*, const char*);
extern int	mimeclose(Mime_t*);
extern int	mimeset(Mime_t*, char*, unsigned long);
extern char*	mimetype(Mime_t*, Sfio_t*, const char*, struct stat*);
extern char*	mimeview(Mime_t*, const char*, const char*, const char*, const char*);
extern int	mimehead(Mime_t*, void*, size_t, size_t, char*);
extern int	mimecmp(const char*, const char*, char**);

#undef	extern

#endif
