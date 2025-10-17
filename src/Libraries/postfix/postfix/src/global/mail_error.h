/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

#ifndef _MAIL_ERROR_H_INCLUDED_
#define _MAIL_ERROR_H_INCLUDED_

/*++
/* NAME
/*	mail_error 3h
/* SUMMARY
/*	mail error classes
/* SYNOPSIS
/*	#include <mail_error.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <name_mask.h>

 /*
  * External interface.
  */
#define MAIL_ERROR_POLICY	(1<<0)
#define MAIL_ERROR_PROTOCOL	(1<<1)
#define MAIL_ERROR_BOUNCE	(1<<2)
#define MAIL_ERROR_SOFTWARE	(1<<3)
#define MAIL_ERROR_RESOURCE	(1<<4)
#define MAIL_ERROR_2BOUNCE	(1<<5)
#define MAIL_ERROR_DELAY	(1<<6)
#define MAIL_ERROR_DATA		(1<<7)

extern const NAME_MASK mail_error_masks[];

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
