/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
/* $Id: fsaccess_test.c,v 1.13 2007/06/19 23:46:59 tbox Exp $ */

/*! \file */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <sys/types.h>		/* Non-portable. */
#include <sys/stat.h>		/* Non-portable. */

#include <isc/fsaccess.h>
#include <isc/print.h>
#include <isc/result.h>

#define PATH "/tmp/fsaccess"

int
main(void) {
	isc_fsaccess_t access;
	isc_result_t result;
	FILE *fp;
	int n;

	n = remove(PATH);
	if (n != 0 && errno != ENOENT) {
		fprintf(stderr, "unable to remove(%s)\n", PATH);
		exit(1);
	}
	fp = fopen(PATH, "w");
	if (fp == NULL) {
		fprintf(stderr, "unable to fopen(%s)\n", PATH);
		exit(1);
	}
	n = chmod(PATH, 0);
	if (n != 0) {
		fprintf(stderr, "unable chmod(%s, 0)\n", PATH);
		exit(1);
	}

	access = 0;

	isc_fsaccess_add(ISC_FSACCESS_OWNER | ISC_FSACCESS_GROUP,
			 ISC_FSACCESS_READ | ISC_FSACCESS_WRITE,
			 &access);

	printf("fsaccess=%d\n", access);

	isc_fsaccess_add(ISC_FSACCESS_OTHER, ISC_FSACCESS_READ, &access);

	printf("fsaccess=%d\n", access);

	result = isc_fsaccess_set(PATH, access);
	if (result != ISC_R_SUCCESS)
		fprintf(stderr, "result = %s\n", isc_result_totext(result));
	(void)fclose(fp);

	return (0);
}
