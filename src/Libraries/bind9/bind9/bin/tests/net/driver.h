/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
/* $Id: driver.h,v 1.8 2007/06/19 23:47:00 tbox Exp $ */

/*
 * PASSED and FAILED mean the particular test passed or failed.
 *
 * UNKNOWN means that for one reason or another, the test process itself
 * failed.  For instance, missing files, error when parsing files or
 * IP addresses, etc.  That is, the test itself is broken, not what is
 * being tested.
 *
 * UNTESTED means the test was unable to be run because a prerequisite test
 * failed, the test is disabled, or the test needs a system component
 * (for instance, Perl) and cannot run.
 */
typedef enum {
	PASSED = 0,
	FAILED = 1,
	UNKNOWN = 2,
	UNTESTED = 3
} test_result_t;

typedef test_result_t (*test_func_t)(void);

typedef struct {
	const char *tag;
	const char *description;
	test_func_t func;
} test_t;

#define TESTDECL(name)	test_result_t name(void)

