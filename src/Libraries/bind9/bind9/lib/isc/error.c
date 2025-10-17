/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
/* $Id: error.c,v 1.21 2007/06/19 23:47:17 tbox Exp $ */

/*! \file */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>

#include <isc/error.h>
#include <isc/msgs.h>
#include <isc/print.h>

/*% Default unexpected callback. */
static void
default_unexpected_callback(const char *, int, const char *, va_list)
     ISC_FORMAT_PRINTF(3, 0);

/*% Default fatal callback. */
static void
default_fatal_callback(const char *, int, const char *, va_list)
     ISC_FORMAT_PRINTF(3, 0);

/*% unexpected_callback */
static isc_errorcallback_t unexpected_callback = default_unexpected_callback;
static isc_errorcallback_t fatal_callback = default_fatal_callback;

void
isc_error_setunexpected(isc_errorcallback_t cb) {
	if (cb == NULL)
		unexpected_callback = default_unexpected_callback;
	else
		unexpected_callback = cb;
}

void
isc_error_setfatal(isc_errorcallback_t cb) {
	if (cb == NULL)
		fatal_callback = default_fatal_callback;
	else
		fatal_callback = cb;
}

void
isc_error_unexpected(const char *file, int line, const char *format, ...) {
	va_list args;

	va_start(args, format);
	(unexpected_callback)(file, line, format, args);
	va_end(args);
}

void
isc_error_fatal(const char *file, int line, const char *format, ...) {
	va_list args;

	va_start(args, format);
	(fatal_callback)(file, line, format, args);
	va_end(args);
	abort();
}

void
isc_error_runtimecheck(const char *file, int line, const char *expression) {
	isc_error_fatal(file, line, "RUNTIME_CHECK(%s) %s", expression,
			isc_msgcat_get(isc_msgcat, ISC_MSGSET_GENERAL,
				       ISC_MSG_FAILED, "failed"));
}

static void
default_unexpected_callback(const char *file, int line, const char *format,
			    va_list args)
{
	fprintf(stderr, "%s:%d: ", file, line);
	vfprintf(stderr, format, args);
	fprintf(stderr, "\n");
	fflush(stderr);
}

static void
default_fatal_callback(const char *file, int line, const char *format,
		       va_list args)
{
	fprintf(stderr, "%s:%d: %s: ", file, line,
		isc_msgcat_get(isc_msgcat, ISC_MSGSET_GENERAL,
			       ISC_MSG_FATALERROR, "fatal error"));
	vfprintf(stderr, format, args);
	fprintf(stderr, "\n");
	fflush(stderr);
}
