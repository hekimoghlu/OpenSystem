/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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

#ifndef _SET_EUGID_H_INCLUDED_
#define _SET_EUGID_H_INCLUDED_

/*++
/* NAME
/*	set_eugid 3h
/* SUMMARY
/*	set effective user and group attributes
/* SYNOPSIS
/*	#include <set_eugid.h>
/* DESCRIPTION
/* .nf

 /* External interface. */

extern void set_eugid(uid_t, gid_t);

 /*
  * The following macros open and close a block that runs at a different
  * privilege level. To make mistakes with stray curly braces less likely, we
  * shape the macros below as the head and tail of a do-while loop.
  */
#define SAVE_AND_SET_EUGID(uid, gid) do { \
	uid_t __set_eugid_uid = geteuid(); \
	gid_t __set_eugid_gid = getegid(); \
	set_eugid((uid), (gid));

#define RESTORE_SAVED_EUGID() \
	set_eugid(__set_eugid_uid, __set_eugid_gid); \
    } while (0)

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
