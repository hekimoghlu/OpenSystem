/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#ifndef ISC_PRINT_H
#define ISC_PRINT_H 1

/*! \file isc/print.h */

/***
 *** Imports
 ***/

#include <isc/formatcheck.h>    /* Required for ISC_FORMAT_PRINTF() macro. */
#include <isc/lang.h>
#include <isc/platform.h>

/*!
 * This block allows lib/isc/print.c to be cleanly compiled even if
 * the platform does not need it.  The standard Makefile will still
 * not compile print.c or archive print.o, so this is just to make test
 * compilation ("make print.o") easier.
 */
#if !defined(ISC_PLATFORM_NEEDVSNPRINTF) && defined(ISC__PRINT_SOURCE)
#define ISC_PLATFORM_NEEDVSNPRINTF
#undef snprintf
#undef vsnprintf
#endif

#if !defined(ISC_PLATFORM_NEEDSPRINTF) && defined(ISC__PRINT_SOURCE)
#define ISC_PLATFORM_NEEDSPRINTF
#undef sprintf
#endif

#if !defined(ISC_PLATFORM_NEEDFPRINTF) && defined(ISC__PRINT_SOURCE)
#define ISC_PLATFORM_NEEDFPRINTF
#undef fprintf
#endif

#if !defined(ISC_PLATFORM_NEEDPRINTF) && defined(ISC__PRINT_SOURCE)
#define ISC_PLATFORM_NEEDPRINTF
#undef printf
#endif

/***
 *** Macros
 ***/
#define ISC_PRINT_QUADFORMAT ISC_PLATFORM_QUADFORMAT

/***
 *** Functions
 ***/

#ifdef ISC_PLATFORM_NEEDVSNPRINTF
#include <stdarg.h>
#include <stddef.h>
#endif

#include <stdio.h>

ISC_LANG_BEGINDECLS

#ifdef ISC_PLATFORM_NEEDVSNPRINTF
int
isc_print_vsnprintf(char *str, size_t size, const char *format, va_list ap)
     ISC_FORMAT_PRINTF(3, 0);
#undef vsnprintf
#define vsnprintf isc_print_vsnprintf

int
isc_print_snprintf(char *str, size_t size, const char *format, ...)
     ISC_FORMAT_PRINTF(3, 4);
#undef snprintf
#define snprintf isc_print_snprintf
#endif /* ISC_PLATFORM_NEEDVSNPRINTF */

#ifdef ISC_PLATFORM_NEEDSPRINTF
int
isc_print_sprintf(char *str, const char *format, ...) ISC_FORMAT_PRINTF(2, 3);
#undef sprintf
#define sprintf isc_print_sprintf
#endif

#ifdef ISC_PLATFORM_NEEDPRINTF
int
isc_print_printf(const char *format, ...) ISC_FORMAT_PRINTF(1, 2);
#undef printf
#define printf isc_print_printf
#endif

#ifdef ISC_PLATFORM_NEEDFPRINTF
int
isc_print_fprintf(FILE * fp, const char *format, ...) ISC_FORMAT_PRINTF(2, 3);
#undef fprintf
#define fprintf isc_print_fprintf
#endif

ISC_LANG_ENDDECLS

#endif /* ISC_PRINT_H */
