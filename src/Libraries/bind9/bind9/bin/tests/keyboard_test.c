/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
/* $Id: keyboard_test.c,v 1.13 2007/06/19 23:46:59 tbox Exp $ */

/*! \file */
#include <config.h>

#include <stdio.h>
#include <stdlib.h>

#include <isc/keyboard.h>
#include <isc/print.h>
#include <isc/util.h>

static void
CHECK(const char *msg, isc_result_t result) {
	if (result != ISC_R_SUCCESS) {
		printf("FAILURE:  %s:  %s\n", msg, isc_result_totext(result));
		exit(1);
	}
}

int
main(int argc, char **argv) {
	isc_keyboard_t kbd;
	unsigned char c;
	isc_result_t res;
	unsigned int count;

	UNUSED(argc);
	UNUSED(argv);

	printf("Type Q to exit.\n");

	res = isc_keyboard_open(&kbd);
	CHECK("isc_keyboard_open()", res);

	c = 'x';
	count = 0;
	while (res == ISC_R_SUCCESS && c != 'Q') {
		res = isc_keyboard_getchar(&kbd, &c);
		printf(".");
		fflush(stdout);
		count++;
		if (count % 64 == 0)
			printf("\r\n");
	}
	printf("\r\n");
	if (res != ISC_R_SUCCESS) {
		printf("FAILURE:  keyboard getchar failed:  %s\r\n",
		       isc_result_totext(res));
		goto errout;
	}

 errout:
	res = isc_keyboard_close(&kbd, 3);
	CHECK("isc_keyboard_close()", res);

	return (0);
}

