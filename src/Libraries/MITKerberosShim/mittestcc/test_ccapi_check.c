/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

#include "test_ccapi_check.h"

int _check_if(int expression, const char *file, int line, const char *expression_string, const char *format, ...) {
	if (expression) {
		failure_count++;
		// call with NULL format to get a generic error message
		if (format == NULL) {
			_log_error(file, line, expression_string, NULL);
		}
		// call with format and varargs for a more useful error message
		else {
			va_list ap;
			va_start(ap, format);
			_log_error_v(file, line, format, ap);
			va_end(ap);
		}
		
		if (current_test_activity) {
			fprintf(stdout, " (%s)", current_test_activity);
		}
	}
	
	return (expression != 0);	
}

int array_contains_int(cc_int32 *array, int size, cc_int32 value) {
	if (array != NULL && size > 0) {
		int i = 0;
		while (i < size && array[i] != value) { 
			i++; 
		}
		if (i < size) {
			return 1;
		}
	}
	return 0;
}
