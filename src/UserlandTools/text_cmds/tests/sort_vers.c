/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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

#include <darwintest.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <limits.h>
 
T_DECL(versionstring, "Apple specific version string") {
	char version[128];
	FILE *sortfile = popen("/usr/bin/sort --version", "r");
	T_ASSERT_NOTNULL(sortfile, "Getting version string");
	T_ASSERT_NOTNULL(fgets(version, sizeof(version), sortfile), "Reading version string");
	pclose(sortfile);
	T_ASSERT_NOTNULL(strstr(version, "-Apple"), "Apple in version string");

	char *num = strstr(version, "(");
	char *endnum = strstr(version, ")");
	T_ASSERT_NOTNULL(num, "Locating parens start");
	T_ASSERT_NOTNULL(endnum, "Locating parens end");
	T_ASSERT_GT(endnum, num, "end is after the start");
	long applevers = strtol(num+1, &endnum, 10);
	T_ASSERT_GT(applevers, 0, "Version greater than zero");
	T_ASSERT_LT(applevers, LONG_MAX, "Version less than LONG_MAX");
}
