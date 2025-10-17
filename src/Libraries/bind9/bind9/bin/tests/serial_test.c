/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
/* $Id: serial_test.c,v 1.15 2007/06/19 23:46:59 tbox Exp $ */

#include <config.h>

#include <stdio.h>

#include <isc/print.h>
#include <isc/serial.h>
#include <isc/stdlib.h>

int
main() {
	isc_uint32_t a, b;
	char buf[1024];
	char *s, *e;

	while (fgets(buf, sizeof(buf), stdin) != NULL) {
		buf[sizeof(buf) - 1] = '\0';
		s = buf;
		a = strtoul(s, &e, 0);
		if (s == e)
			continue;
		s = e;
		b = strtoul(s, &e, 0);
		if (s == e)
			continue;
		fprintf(stdout, "%u %u gt:%d lt:%d ge:%d le:%d eq:%d ne:%d\n",
			a, b,
			isc_serial_gt(a,b), isc_serial_lt(a,b),
			isc_serial_ge(a,b), isc_serial_le(a,b),
			isc_serial_eq(a,b), isc_serial_ne(a,b));
	}
	return (0);
}
