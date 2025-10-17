/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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

#ifndef _USER_ACL_H_INCLUDED_
#define _USER_ACL_H_INCLUDED_
/*++
/* NAME
/*	user_acl 3h
/* SUMMARY
/*	Convert uid to username and check against given ACL.
/* SYNOPSIS
/*	#include <user_acl.h>
/*
/* DESCRIPTION
/* .nf

 /*
  * System library
  */
#include <unistd.h>		/* getuid()/geteuid() */
#include <sys/types.h>		/* uid_t */

 /*
  * Utility library.
  */
#include <vstring.h>

 /*
  * External interface
  */
extern const char *check_user_acl_byuid(const char *, const char *, uid_t);

/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*
/*	Victor Duchovni
/*	Morgan Stanley
/*--*/
#endif
