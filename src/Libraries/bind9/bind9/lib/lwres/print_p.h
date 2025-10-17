/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
/* $Id: print_p.h,v 1.6 2010/08/16 23:46:52 tbox Exp $ */

#ifndef LWRES_PRINT_P_H
#define LWRES_PRINT_P_H 1

/***
 *** Imports
 ***/

#include <lwres/lang.h>
#include <lwres/platform.h>

/*
 * This block allows lib/lwres/print.c to be cleanly compiled even if
 * the platform does not need it.  The standard Makefile will still
 * not compile print.c or archive print.o, so this is just to make test
 * compilation ("make print.o") easier.
 */
#if !defined(LWRES_PLATFORM_NEEDVSNPRINTF) && defined(LWRES__PRINT_SOURCE)
#define LWRES_PLATFORM_NEEDVSNPRINTF
#endif

#if !defined(LWRES_PLATFORM_NEEDSPRINTF) && defined(LWRES__PRINT_SOURCE)
#define LWRES_PLATFORM_NEEDSPRINTF
#endif

/***
 *** Macros.
 ***/

#ifdef __GNUC__
#define LWRES_FORMAT_PRINTF(fmt, args) \
	__attribute__((__format__(__printf__, fmt, args)))
#else
#define LWRES_FORMAT_PRINTF(fmt, args)
#endif

/***
 *** Functions
 ***/

#ifdef LWRES_PLATFORM_NEEDVSNPRINTF
#include <stdarg.h>
#include <stddef.h>
#endif

LWRES_LANG_BEGINDECLS

#ifdef LWRES_PLATFORM_NEEDVSNPRINTF
int
lwres__print_vsnprintf(char *str, size_t size, const char *format, va_list ap)
     LWRES_FORMAT_PRINTF(3, 0);
#ifdef vsnprintf
#undef vsnprintf
#endif
#define vsnprintf lwres__print_vsnprintf

int
lwres__print_snprintf(char *str, size_t size, const char *format, ...)
     LWRES_FORMAT_PRINTF(3, 4);
#ifdef snprintf
#undef snprintf
#endif
#define snprintf lwres__print_snprintf
#endif /* LWRES_PLATFORM_NEEDVSNPRINTF */

#ifdef LWRES_PLATFORM_NEEDSPRINTF
int
lwres__print_sprintf(char *str, const char *format, ...) LWRES_FORMAT_PRINTF(2, 3);
#ifdef sprintf
#undef sprintf
#endif
#define sprintf lwres__print_sprintf
#endif

LWRES_LANG_ENDDECLS

#endif /* LWRES_PRINT_P_H */
