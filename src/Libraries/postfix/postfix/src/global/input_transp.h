/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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

#ifndef _INPUT_TRANSP_INCLUDED_
#define _INPUT_TRANSP_INCLUDED_

/*++
/* NAME
/*	input_transp 3h
/* SUMMARY
/*	receive transparency control
/* SYNOPSIS
/*	#include <input_transp.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
#define INPUT_TRANSP_UNKNOWN_RCPT	(1<<0)
#define INPUT_TRANSP_ADDRESS_MAPPING	(1<<1)
#define INPUT_TRANSP_HEADER_BODY	(1<<2)
#define INPUT_TRANSP_MILTER		(1<<3)

extern int input_transp_mask(const char *, const char *);
extern int input_transp_cleanup(int, int);

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
