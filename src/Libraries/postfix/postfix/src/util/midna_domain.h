/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#ifndef _MIDNA_H_INCLUDED_
#define _MIDNA_H_INCLUDED_

/*++
/* NAME
/*	midna_domain 3h
/* SUMMARY
/*	ASCII/UTF-8 domain name conversion
/* SYNOPSIS
/*	#include <midna_domain.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
extern const char *midna_domain_to_ascii(const char *);
extern const char *midna_domain_to_utf8(const char *);
extern const char *midna_domain_suffix_to_ascii(const char *);
extern const char *midna_domain_suffix_to_utf8(const char *);

extern int midna_domain_cache_size;
extern int midna_domain_transitional;
/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Arnt Gulbrandsen
/*
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*
/*	Wietse Venema
/*	Google, Inc.
/*	111 8th Avenue
/*	New York, NY 10011, USA
/*--*/

#endif
