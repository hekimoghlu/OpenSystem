/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
/* $Id: util.h,v 1.4 2009/09/29 15:06:05 fdupont Exp $ */

#ifndef RNDC_UTIL_H
#define RNDC_UTIL_H 1

/*! \file */

#include <isc/lang.h>
#include <isc/platform.h>

#include <isc/formatcheck.h>

#define NS_CONTROL_PORT		953

#undef DO
#define DO(name, function) \
	do { \
		result = function; \
		if (result != ISC_R_SUCCESS) \
			fatal("%s: %s", name, isc_result_totext(result)); \
		else \
			notify("%s", name); \
	} while (0)

ISC_LANG_BEGINDECLS

void
notify(const char *fmt, ...) ISC_FORMAT_PRINTF(1, 2);

ISC_PLATFORM_NORETURN_PRE void
fatal(const char *format, ...)
ISC_FORMAT_PRINTF(1, 2) ISC_PLATFORM_NORETURN_POST;

ISC_LANG_ENDDECLS

#endif /* RNDC_UTIL_H */
