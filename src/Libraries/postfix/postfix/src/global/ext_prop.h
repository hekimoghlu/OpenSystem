/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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

#ifndef _EXT_PROP_INCLUDED_
#define _EXT_PROP_INCLUDED_

/*++
/* NAME
/*	ext_prop 3h
/* SUMMARY
/*	address extension propagation control
/* SYNOPSIS
/*	#include <ext_prop.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
#define EXT_PROP_CANONICAL	(1<<0)
#define EXT_PROP_VIRTUAL	(1<<1)
#define EXT_PROP_ALIAS		(1<<2)
#define EXT_PROP_FORWARD	(1<<3)
#define EXT_PROP_INCLUDE	(1<<4)
#define EXT_PROP_GENERIC	(1<<5)

extern int ext_prop_mask(const char *, const char *);

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
