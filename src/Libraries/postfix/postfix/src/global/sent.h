/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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

#ifndef _SENT_H_INCLUDED_
#define _SENT_H_INCLUDED_

/*++
/* NAME
/*	sent 3h
/* SUMMARY
/*	log that message was sent
/* SYNOPSIS
/*	#include <sent.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <time.h>
#include <stdarg.h>

 /*
  * Global library.
  */
#include <deliver_request.h>
#include <bounce.h>

 /*
  * External interface.
  */
#define SENT_FLAG_NONE	(0)

extern int sent(int, const char *, MSG_STATS *, RECIPIENT *, const char *, 
			DSN *);

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
