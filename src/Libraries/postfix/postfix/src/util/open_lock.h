/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#ifndef _OPEN_LOCK_H_INCLUDED_
#define _OPEN_LOCK_H_INCLUDED_

/*++
/* NAME
/*	open_lock 3h
/* SUMMARY
/*	open or create file and lock it for exclusive access
/* SYNOPSIS
/*	#include <open_lock.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <fcntl.h>

 /*
  * Utility library.
  */
#include <vstream.h>
#include <vstring.h>

 /*
  * External interface.
  */
extern VSTREAM *open_lock(const char *, int, mode_t, VSTRING *);

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
