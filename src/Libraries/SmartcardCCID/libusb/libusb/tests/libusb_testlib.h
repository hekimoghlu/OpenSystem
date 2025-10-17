/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#ifndef LIBUSB_TESTLIB_H
#define LIBUSB_TESTLIB_H

#include <config.h>

/** Values returned from a test function to indicate test result */
typedef enum {
	/** Indicates that the test ran successfully. */
	TEST_STATUS_SUCCESS,
	/** Indicates that the test failed one or more test. */
	TEST_STATUS_FAILURE,
	/** Indicates that an unexpected error occurred. */
	TEST_STATUS_ERROR,
	/** Indicates that the test can't be run. For example this may be
	 * due to no suitable device being connected to perform the tests. */
	TEST_STATUS_SKIP
} libusb_testlib_result;

/**
 * Logs some test information or state
 */
void libusb_testlib_logf(const char *fmt, ...) PRINTF_FORMAT(1, 2);

/**
 * Structure holding a test description.
 */
typedef struct {
	/** Human readable name of the test. */
	const char *name;
	/** The test library will call this function to run the test.
	 *
	 * Should return TEST_STATUS_SUCCESS on success or another TEST_STATUS value.
	 */
	libusb_testlib_result (*function)(void);
} libusb_testlib_test;

/**
 * Value to use at the end of a test array to indicate the last
 * element.
 */
#define LIBUSB_NULL_TEST { NULL, NULL }

/**
 * Runs the tests provided.
 *
 * Before running any tests argc and argv will be processed
 * to determine the mode of operation.
 *
 * \param argc The argc from main
 * \param argv The argv from main
 * \param tests A NULL_TEST terminated array of tests
 * \return 0 on success, non-zero on failure
 */
int libusb_testlib_run_tests(int argc, char *argv[],
	const libusb_testlib_test *tests);

#endif //LIBUSB_TESTLIB_H
