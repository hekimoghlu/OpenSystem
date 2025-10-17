/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
/*
 * @OSF_FREE_COPYRIGHT@
 *
 */

#include <mach/mach.h>
#include <mach/mach_init.h>
#include <stdarg.h>
#include "string.h"

int (*vprintf_stderr_func)(const char *format, va_list ap);

#define __STDERR_FILENO 2
int write(int fd, const char* cbuf, int nbyte);

/* This function allows the writing of a mach error message to an
 * application-controllable output method, the default being to
 * use printf if no other method is specified by the application.
 *
 * To override, set the global function pointer vprintf_stderr to
 * a function which takes the same parameters as vprintf.
 */

__private_extern__ int
fprintf_stderr(const char *format, ...)
{
	va_list args;
	int retval;

	va_start(args, format);
	if (vprintf_stderr_func == NULL) {
		char buffer[1024];
		retval = _mach_vsnprintf(buffer, sizeof(buffer), format, args);
		write(__STDERR_FILENO, buffer, retval);
	} else {
		retval = (*vprintf_stderr_func)(format, args);
	}
	va_end(args);

	return retval;
}
