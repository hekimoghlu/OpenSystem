/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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

#ifndef _TEST_CCAPI_LOG_H_
#define _TEST_CCAPI_LOG_H_

#include <stdio.h>
#include <stdarg.h>
#include "test_ccapi_globals.h"

#define log_error(format, ...) \
		_log_error(__FILE__, __LINE__, format , ## __VA_ARGS__)

void _log_error_v(const char *file, int line, const char *format, va_list ap);
void _log_error(const char *file, int line, const char *format, ...)
#if __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)
__attribute__ ((__format__ (__printf__, 3, 4)))
#endif
;
void test_header(const char *msg);
void test_footer(const char *msg, int err);

#endif /* _TEST_CCAPI_LOG_H_ */
