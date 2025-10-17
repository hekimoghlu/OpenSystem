/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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

#ifndef _MAIL_ADDR_H_INCLUDED_
#define _MAIL_ADDR_H_INCLUDED_

/*++
/* NAME
/*	mail_addr 3h
/* SUMMARY
/*	pre-defined mail addresses
/* SYNOPSIS
/*	#include <mail_addr.h>
/* DESCRIPTION
/* .nf

 /*
  * Pre-defined addresses.
  */
#define MAIL_ADDR_POSTMASTER	"postmaster"
#define MAIL_ADDR_MAIL_DAEMON	"MAILER-DAEMON"
#define MAIL_ADDR_EMPTY		""

extern const char *mail_addr_double_bounce(void);
extern const char *mail_addr_postmaster(void);
extern const char *mail_addr_mail_daemon(void);

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
