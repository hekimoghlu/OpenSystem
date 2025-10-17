/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
/* $Id: backtrace_test.c,v 1.4 2009/09/02 23:48:01 tbox Exp $ */

#include <config.h>

#include <stdio.h>
#include <string.h>

#include <isc/backtrace.h>
#include <isc/print.h>
#include <isc/result.h>

const char *expected_symbols[] = {
	"func3",
	"func2",
	"func1",
	"main"
};

static int
func3() {
	void *tracebuf[16];
	int i, nframes;
	int error = 0;
	const char *fname;
	isc_result_t result;
	unsigned long offset;

	result = isc_backtrace_gettrace(tracebuf, 16, &nframes);
	if (result != ISC_R_SUCCESS) {
		printf("isc_backtrace_gettrace failed: %s\n",
		       isc_result_totext(result));
		return (1);
	}

	if (nframes < 4)
		error++;

	for (i = 0; i < 4 && i < nframes; i++) {
		fname = NULL;
		result = isc_backtrace_getsymbol(tracebuf[i], &fname, &offset);
		if (result != ISC_R_SUCCESS) {
			error++;
			continue;
		}
		if (strcmp(fname, expected_symbols[i]) != 0)
			error++;
	}

	if (error) {
		printf("Unexpected result:\n");
		printf("  # of frames: %d (expected: at least 4)\n", nframes);
		printf("  symbols:\n");
		for (i = 0; i < nframes; i++) {
			fname = NULL;
			result = isc_backtrace_getsymbol(tracebuf[i], &fname,
							 &offset);
			if (result == ISC_R_SUCCESS)
				printf("  [%d] %s\n", i, fname);
			else {
				printf("  [%d] %p getsymbol failed: %s\n", i,
				       tracebuf[i], isc_result_totext(result));
			}
		}
	}

	return (error);
}

static int
func2() {
	return (func3());
}

static int
func1() {
	return (func2());
}

int
main() {
	return (func1());
}
