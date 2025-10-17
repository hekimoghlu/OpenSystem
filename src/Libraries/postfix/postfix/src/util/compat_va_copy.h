/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

#ifndef _COMPAT_VA_COPY_H_INCLUDED_
#define _COMPAT_VA_COPY_H_INCLUDED_

/*++
/* NAME
/*	compat_va_copy 3h
/* SUMMARY
/*	compatibility
/* SYNOPSIS
/*	#include <compat_va_copy.h>
/* DESCRIPTION
/* .nf

 /*
  * C99 defines va_start and va_copy as macros, so we can probe the
  * compilation environment with #ifdef etc. Some environments define
  * __va_copy so we probe for that, too.
  */
#if !defined(va_start)
#error "include <stdarg.h> first"
#endif

#if !defined(VA_COPY)
#if defined(va_copy)
#define VA_COPY(dest, src) va_copy(dest, src)
#elif defined(__va_copy)
#define VA_COPY(dest, src) __va_copy(dest, src)
#else
#define VA_COPY(dest, src) (dest) = (src)
#endif
#endif					/* VA_COPY */

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
