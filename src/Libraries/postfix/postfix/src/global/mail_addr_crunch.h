/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

#ifndef _MAIL_ADDR_CRUNCH_H_INCLUDED_
#define _MAIL_ADDR_CRUNCH_H_INCLUDED_

/*++
/* NAME
/*	mail_addr_crunch 3h
/* SUMMARY
/*	parse and canonicalize addresses, apply address extension
/* SYNOPSIS
/*	#include <mail_addr_crunch.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <argv.h>

 /*
  * Global library.
  */
#include <mail_addr_form.h>

 /*
  * External interface.
  */
extern ARGV *mail_addr_crunch_opt(const char *, const char *, int, int);

 /*
  * The least-overhead form.
  */
#define mail_addr_crunch_ext_to_int(string, extension) \
	mail_addr_crunch_opt((string), (extension), MA_FORM_EXTERNAL, \
			MA_FORM_INTERNAL)

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
