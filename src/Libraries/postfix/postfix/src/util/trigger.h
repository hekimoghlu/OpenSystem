/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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

#ifndef _TRIGGER_H_INCLUDED_
#define _TRIGGER_H_INCLUDED_

/*++
/* NAME
/*	trigger 3h
/* SUMMARY
/*	client interface file
/* SYNOPSIS
/*	#include <trigger.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
extern int unix_trigger(const char *, const char *, ssize_t, int);
extern int inet_trigger(const char *, const char *, ssize_t, int);
extern int fifo_trigger(const char *, const char *, ssize_t, int);
extern int stream_trigger(const char *, const char *, ssize_t, int);
extern int pass_trigger(const char *, const char *, ssize_t, int);

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
