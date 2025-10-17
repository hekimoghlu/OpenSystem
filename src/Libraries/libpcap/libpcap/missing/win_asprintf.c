/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "portability.h"

int
pcap_vasprintf(char **strp, const char *format, va_list args)
{
	int len;
	size_t str_size;
	char *str;
	int ret;

	len = _vscprintf(format, args);
	if (len == -1) {
		*strp = NULL;
		return (-1);
	}
	str_size = len + 1;
	str = malloc(str_size);
	if (str == NULL) {
		*strp = NULL;
		return (-1);
	}
	ret = vsnprintf(str, str_size, format, args);
	if (ret == -1) {
		free(str);
		*strp = NULL;
		return (-1);
	}
	*strp = str;
	/*
	 * vsnprintf() shouldn't truncate the string, as we have
	 * allocated a buffer large enough to hold the string, so its
	 * return value should be the number of characters printed.
	 */
	return (ret);
}

int
pcap_asprintf(char **strp, const char *format, ...)
{
	va_list args;
	int ret;

	va_start(args, format);
	ret = pcap_vasprintf(strp, format, args);
	va_end(args);
	return (ret);
}
