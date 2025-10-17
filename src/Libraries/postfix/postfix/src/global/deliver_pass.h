/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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

#ifndef _DELIVER_PASS_H_INCLUDED_
#define _DELIVER_PASS_H_INCLUDED_

/*++
/* NAME
/*	deliver_pass 3h
/* SUMMARY
/*	deliver request pass_through
/* SYNOPSIS
/*	#include <deliver_pass.h>
/* DESCRIPTION
/* .nf

 /*
  * Global library.
  */
#include <deliver_request.h>
#include <mail_proto.h>

 /*
  * External interface.
  */
extern int deliver_pass(const char *, const char *, DELIVER_REQUEST *, RECIPIENT *);
extern int deliver_pass_all(const char *, const char *, DELIVER_REQUEST *);

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
