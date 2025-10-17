/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#include <config.h>

#include <string.h>

#include "libusb.h"
#include "libusb_testlib.h"

/** Test that creates and destroys a single concurrent context
 * 10000 times. */
static libusb_testlib_result test_init_and_exit(void)
{
	for (int i = 0; i < 10000; ++i) {
		libusb_context *ctx = NULL;
		int r;

		r = libusb_init(&ctx);
		if (r != LIBUSB_SUCCESS) {
			libusb_testlib_logf(
				"Failed to init libusb on iteration %d: %d",
				i, r);
			return TEST_STATUS_FAILURE;
		}
		libusb_exit(ctx);
	}

	return TEST_STATUS_SUCCESS;
}

/** Tests that devices can be listed 1000 times. */
static libusb_testlib_result test_get_device_list(void)
{
	libusb_context *ctx;
	int r;

	r = libusb_init(&ctx);
	if (r != LIBUSB_SUCCESS) {
		libusb_testlib_logf("Failed to init libusb: %d", r);
		return TEST_STATUS_FAILURE;
	}

	for (int i = 0; i < 1000; ++i) {
		libusb_device **device_list = NULL;
		ssize_t list_size = libusb_get_device_list(ctx, &device_list);
		if (list_size < 0 || !device_list) {
			libusb_testlib_logf(
				"Failed to get device list on iteration %d: %ld (%p)",
				i, (long)-list_size, device_list);
			libusb_exit(ctx);
			return TEST_STATUS_FAILURE;
		}
		libusb_free_device_list(device_list, 1);
	}

	libusb_exit(ctx);
	return TEST_STATUS_SUCCESS;
}

/** Tests that 100 concurrent device lists can be open at a time. */
static libusb_testlib_result test_many_device_lists(void)
{
#define LIST_COUNT 100
	libusb_testlib_result result = TEST_STATUS_SUCCESS;
	libusb_context *ctx = NULL;
	libusb_device **device_lists[LIST_COUNT];
	int r;

	r = libusb_init(&ctx);
	if (r != LIBUSB_SUCCESS) {
		libusb_testlib_logf("Failed to init libusb: %d", r);
		return TEST_STATUS_FAILURE;
	}

	memset(device_lists, 0, sizeof(device_lists));

	/* Create the 100 device lists. */
	for (int i = 0; i < LIST_COUNT; ++i) {
		ssize_t list_size = libusb_get_device_list(ctx, &device_lists[i]);
		if (list_size < 0 || !device_lists[i]) {
			libusb_testlib_logf(
				"Failed to get device list on iteration %d: %ld (%p)",
				i, (long)-list_size, device_lists[i]);
			result = TEST_STATUS_FAILURE;
			break;
		}
	}

	/* Destroy the 100 device lists. */
	for (int i = 0; i < LIST_COUNT; ++i) {
		if (device_lists[i])
			libusb_free_device_list(device_lists[i], 1);
	}

	libusb_exit(ctx);
	return result;
#undef LIST_COUNT
}

/** Tests that the default context (used for various things including
 * logging) works correctly when the first context created in a
 * process is destroyed. */
static libusb_testlib_result test_default_context_change(void)
{
	for (int i = 0; i < 100; ++i) {
		libusb_context *ctx = NULL;
		int r;

		/* First create a new context */
		r = libusb_init(&ctx);
		if (r != LIBUSB_SUCCESS) {
			libusb_testlib_logf("Failed to init libusb: %d", r);
			return TEST_STATUS_FAILURE;
		}

		/* Enable debug output, to be sure to use the context */
		libusb_set_option(NULL, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_DEBUG);
		libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_DEBUG);

		/* Now create a reference to the default context */
		r = libusb_init(NULL);
		if (r != LIBUSB_SUCCESS) {
			libusb_testlib_logf("Failed to init libusb: %d", r);
			libusb_exit(ctx);
			return TEST_STATUS_FAILURE;
		}

		/* Destroy the first context */
		libusb_exit(ctx);
		/* Destroy the default context */
		libusb_exit(NULL);
	}

	return TEST_STATUS_SUCCESS;
}

/* Fill in the list of tests. */
static const libusb_testlib_test tests[] = {
	{ "init_and_exit", &test_init_and_exit },
	{ "get_device_list", &test_get_device_list },
	{ "many_device_lists", &test_many_device_lists },
	{ "default_context_change", &test_default_context_change },
	LIBUSB_NULL_TEST
};

int main(int argc, char *argv[])
{
	return libusb_testlib_run_tests(argc, argv, tests);
}
