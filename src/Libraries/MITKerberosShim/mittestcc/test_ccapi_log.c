/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

#ifndef _TEST_CCAPI_LOG_C_
#define _TEST_CCAPI_LOG_C_

#include "test_ccapi_log.h"

void _log_error_v(const char *file, int line, const char *format, va_list ap)
{
	fprintf(stdout, "\n\t%s:%d: ", file, line);
	if (!format) {
		fprintf(stdout, "An unknown error occurred");
	} else {
		vfprintf(stdout, format, ap);
	}
	fflush(stdout);
}

void _log_error(const char *file, int line, const char *format, ...)
{	
	va_list ap;
	va_start(ap, format);
	_log_error_v(file, line, format, ap);
	va_end(ap);
}

void test_header(const char *msg) {
	if (msg != NULL) {
		fprintf(stdout, "\nChecking %s... ", msg);
		fflush(stdout);
	}
}

void test_footer(const char *msg, int err) {
	if (msg != NULL) {
		if (!err) {
			fprintf(stdout, "OK\n");
		}
		else {
			fprintf(stdout, "\n*** %d failure%s in %s ***\n", err, (err == 1) ? "" : "s", msg);
		}		
	}
}



#endif /* _TEST_CCAPI_LOG_C_ */
