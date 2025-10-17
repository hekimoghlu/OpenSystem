/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
/* $Id: error.h,v 1.22 2009/09/29 23:48:04 tbox Exp $ */

#ifndef ISC_ERROR_H
#define ISC_ERROR_H 1

/*! \file isc/error.h */

#include <stdarg.h>

#include <isc/formatcheck.h>
#include <isc/lang.h>
#include <isc/platform.h>

ISC_LANG_BEGINDECLS

typedef void (*isc_errorcallback_t)(const char *, int, const char *, va_list);

/*% set unexpected error */
void
isc_error_setunexpected(isc_errorcallback_t);

/*% set fatal error */
void
isc_error_setfatal(isc_errorcallback_t);

/*% unexpected error */
void
isc_error_unexpected(const char *, int, const char *, ...)
     ISC_FORMAT_PRINTF(3, 4);

/*% fatal error */
ISC_PLATFORM_NORETURN_PRE void
isc_error_fatal(const char *, int, const char *, ...)
ISC_FORMAT_PRINTF(3, 4) ISC_PLATFORM_NORETURN_POST;

/*% runtimecheck error */
void
isc_error_runtimecheck(const char *, int, const char *);

#define ISC_ERROR_RUNTIMECHECK(cond) \
	((void) (ISC_LIKELY(cond) || \
		 ((isc_error_runtimecheck)(__FILE__, __LINE__, #cond), 0)))

ISC_LANG_ENDDECLS

#endif /* ISC_ERROR_H */
