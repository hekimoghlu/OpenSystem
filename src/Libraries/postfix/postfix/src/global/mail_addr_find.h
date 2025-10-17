/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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

#ifndef _MAIL_ADDR_FIND_H_INCLUDED_
#define _MAIL_ADDR_FIND_H_INCLUDED_

/*++
/* NAME
/*	mail_addr_find 3h
/* SUMMARY
/*	generic address-based lookup
/* SYNOPSIS
/*	#include <mail_addr_find.h>
/* DESCRIPTION
/* .nf

 /*
  * Global library.
  */
#include <mail_addr_form.h>
#include <maps.h>

 /*
  * External interface.
  */
extern const char *mail_addr_find_opt(MAPS *, const char *, char **,
				              int, int, int, int);

#define MA_FIND_FULL	(1<<0)		/* localpart+ext@domain */
#define MA_FIND_NOEXT	(1<<1)		/* localpart@domain */
#define MA_FIND_LOCALPART_IF_LOCAL \
				(1<<2)	/* localpart (maybe localpart+ext) */
#define MA_FIND_LOCALPART_AT_IF_LOCAL \
				(1<<3)	/* ditto, with @ at end */
#define MA_FIND_AT_DOMAIN	(1<<4)	/* @domain */
#define MA_FIND_DOMAIN	(1<<5)		/* domain */
#define MA_FIND_PDMS	(1<<6)		/* parent matches subdomain */
#define MA_FIND_PDDMDS	(1<<7)		/* parent matches dot-subdomain */
#define MA_FIND_LOCALPART_AT	\
				(1<<8)	/* localpart@ (maybe localpart+ext@) */

#define MA_FIND_DEFAULT	(MA_FIND_FULL | MA_FIND_NOEXT \
				| MA_FIND_LOCALPART_IF_LOCAL \
				| MA_FIND_AT_DOMAIN)

 /* The least-overhead form. */
#define mail_addr_find_int_to_ext(maps, address, extension) \
	mail_addr_find_opt((maps), (address), (extension), \
	    MA_FORM_INTERNAL, MA_FORM_EXTERNAL, \
	    MA_FORM_EXTERNAL, MA_FIND_DEFAULT)

 /* The legacy forms. */
#define MA_FIND_FORM_LEGACY \
	MA_FORM_INTERNAL, MA_FORM_EXTERNAL_FIRST, \
	    MA_FORM_EXTERNAL

#define mail_addr_find_strategy(maps, address, extension, strategy) \
	mail_addr_find_opt((maps), (address), (extension), \
	    MA_FIND_FORM_LEGACY, (strategy))

#define mail_addr_find(maps, address, extension) \
	mail_addr_find_strategy((maps), (address), (extension), \
	    MA_FIND_DEFAULT)

#define mail_addr_find_to_internal(maps, address, extension) \
	mail_addr_find_opt((maps), (address), (extension), \
	    MA_FIND_FORM_LEGACY, MA_FIND_DEFAULT)

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
