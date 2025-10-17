/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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

#ifndef _MAIL_ADDR_FORM_H_INCLUDED_
#define _MAIL_ADDR_FORM_H_INCLUDED_

/*++
/* NAME
/*	mail_addr_form 3h
/* SUMMARY
/*	mail address formats
/* SYNOPSIS
/*	#include <mail_addr_form.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
#define MA_FORM_INTERNAL	1	/* unquoted form */
#define MA_FORM_EXTERNAL	2	/* quoted form */
#define MA_FORM_EXTERNAL_FIRST 3	/* quoted form, then unquoted */
#define MA_FORM_INTERNAL_FIRST 4	/* unquoted form, then quoted */

extern int mail_addr_form_from_string(const char *);
extern const char *mail_addr_form_to_string(int);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	Google, Inc.
/*	111 8th Avenue
/*	New York, NY 10011, USA
/*--*/

#endif
