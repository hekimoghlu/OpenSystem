/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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
/*
 * Implementation and tests of thread test contexts.
 */

#if !(DEBUG || DEVELOPMENT)
#error this file is not for release
#endif

#include <kern/thread_test_context.h>

/* For testing thread_test_context_t itself. */
DECLARE_TEST_IDENTITY(test_identity_thread_test_context);
DEFINE_TEST_IDENTITY(test_identity_thread_test_context);

void
thread_test_context_deinit(thread_test_context_t *ctx)
{
	/*
	 * Deinitialize thread_text_context_t->ttc_* fields.
	 * Don't touch ttc->ttc_data.
	 */

	/*
	 * for testing ttc itself: modify *ttc->ttc_data so the
	 * test can verify that this deinit was executed.
	 */
	if (ctx->ttc_identity == test_identity_thread_test_context) {
		int *data_p = (int *)ctx->ttc_data;
		if (data_p) {
			*data_p += 1;
		}
	}
}

/* Tests of thread test contexts */

#define FAIL                                    \
	({                                      \
	        *out_value = __LINE__;          \
	        return 0;                       \
	})

static int
thread_test_context_tests(int64_t in_value __unused, int64_t *out_value)
{
	*out_value = 0;

	/*
	 * Tests of:
	 * thread_set_test_context
	 * thread_cleanup_test_context when thread's context is NULL
	 * thread_cleanup_test_context when thread's context is not NULL
	 * thread_test_context_deinit
	 */
	{
		/* no attribute(cleanup), we call cleanup manually */
		int data;
		thread_test_context_t ctx = {
			.ttc_identity = test_identity_thread_test_context,
			.ttc_data = &data,
		};

		data = 0;
		/* cleanup called when thread's context is NULL */
		if (current_thread()->th_test_ctx != NULL) {
			FAIL;
		}
		if (thread_get_test_context() != NULL) {
			FAIL;
		}
		thread_cleanup_test_context(&ctx);
		/* thread_test_context_deinit increments *ttc_data */
		if (data != 1) {
			FAIL;
		}
		/* thread_cleanup_test_context clears thread's context */
		if (current_thread()->th_test_ctx != NULL) {
			FAIL;
		}

		data = 1;
		/* cleanup called when thread's context is not NULL */
		thread_set_test_context(&ctx);
		if (current_thread()->th_test_ctx != &ctx) {
			FAIL;
		}
		if (thread_get_test_context() != &ctx) {
			FAIL;
		}
		thread_cleanup_test_context(&ctx);
		/* thread_test_context_deinit increments *ttc_data */
		if (data != 2) {
			FAIL;
		}
		/* thread_cleanup_test_context clears thread's context */
		if (current_thread()->th_test_ctx != NULL) {
			FAIL;
		}
	}

	/*
	 * Tests of:
	 * access test options with no test context set
	 * access test options when a context is installed but no options are set
	 * attribute(cleanup(thread_cleanup_test_context))
	 */
	int data = 0;
	{
		thread_test_context_t ctx CLEANUP_THREAD_TEST_CONTEXT = {
			.ttc_identity = test_identity_thread_test_context,
			.ttc_data = &data,
			.ttc_testing_ttc_int = 1,
			.ttc_testing_ttc_struct = { 33, 44 }
		};

		/* access test options with no test context set */
		if (thread_get_test_context() != NULL) {
			FAIL;
		}

		if (thread_get_test_option(ttc_testing_ttc_int) != 0) {
			FAIL;
		}
		/* setting an option with no context has no effect */
		thread_set_test_option(ttc_testing_ttc_int, 1 + thread_get_test_option(ttc_testing_ttc_int));
		if (thread_get_test_option(ttc_testing_ttc_int) != 0) {
			FAIL;
		}

		if (thread_get_test_option(ttc_testing_ttc_struct).min_address != 0) {
			FAIL;
		}
		/* setting an option with no context has no effect */
		thread_set_test_option(ttc_testing_ttc_struct, (struct mach_vm_range){55, 66});
		if (thread_get_test_option(ttc_testing_ttc_struct).min_address != 0) {
			FAIL;
		}

		/* access test options with a test context set */
		thread_set_test_context(&ctx);
		if (thread_get_test_option(ttc_testing_ttc_int) != 1) {
			FAIL;
		}
		thread_set_test_option(ttc_testing_ttc_int, 1 + thread_get_test_option(ttc_testing_ttc_int));
		if (thread_get_test_option(ttc_testing_ttc_int) != 2) {
			FAIL;
		}
		thread_set_test_option(ttc_testing_ttc_int, 0);
		if (thread_get_test_option(ttc_testing_ttc_int) != 0) {
			FAIL;
		}

		if (thread_get_test_option(ttc_testing_ttc_struct).min_address != 33) {
			FAIL;
		}
		thread_set_test_option(ttc_testing_ttc_struct, (struct mach_vm_range){55, 66});
		if (thread_get_test_option(ttc_testing_ttc_struct).min_address != 55) {
			FAIL;
		}

		/* thread_cleanup_test_context runs at end of scope */
		if (data != 0) {
			FAIL;
		}
	}
	/* thread_cleanup_test_context incremented data through ttc->ttc_data */
	if (data != 1) {
		FAIL;
	}

	if (current_thread()->th_test_ctx != NULL) {
		FAIL;
	}

	/* success */
	*out_value = 0;
	return 0;
}

#undef FAIL

SYSCTL_TEST_REGISTER(thread_test_context, thread_test_context_tests);
