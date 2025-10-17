/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
#include "internal.h"

#pragma mark Utilities
static void
_print_preamble(FILE *fp, const char *fmt, va_list ap)
{
	fprintf(fp, "%s: ", getprogname());
	vfprintf(fp, fmt, ap);
}

#pragma mark API
void
err_np(errno_t code, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	verr_np(code, fmt, ap);
	va_end(ap);
}

void
errc_np(int eval, errno_t code, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	verrc_np(eval, code, fmt, ap);
	va_end(ap);
}

void
warn_np(errno_t code, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vwarn_np(code, fmt, ap);
	va_end(ap);
}

void
verr_np(errno_t code, const char *fmt, va_list ap)
{
	_print_preamble(stderr, fmt, ap);
	fprintf(stderr, ": %s\n", strerror_np(code));
	exit(sysexit_np(code));
}

void
verrc_np(int eval, errno_t code, const char *fmt, va_list ap)
{
	_print_preamble(stderr, fmt, ap);
	fprintf(stderr, ": %s\n", strerror_np(code));
	exit(eval);
}

void
vwarn_np(errno_t code, const char *fmt, va_list ap)
{
	_print_preamble(stderr, fmt, ap);
	fprintf(stderr, ": %s\n", strerror_np(code));
}
