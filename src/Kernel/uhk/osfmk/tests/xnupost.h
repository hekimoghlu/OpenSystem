/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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
#ifndef _TESTS_XNUPOST_H
#define _TESTS_XNUPOST_H

#ifndef CONFIG_XNUPOST
#error "Testing is not enabled if CONFIG_XNUPOST is not enabled"
#endif

#include <kern/kern_types.h>
#include <kern/assert.h>
#include <tests/ktest.h>

#define XT_CONFIG_RUN 0x0
#define XT_CONFIG_IGNORE 0x1
#define XT_CONFIG_EXPECT_PANIC 0x2

#define XTCTL_RUN_TESTS  1
#define XTCTL_RESET_TESTDATA 2

typedef enum { XT_ACTION_NONE = 0, XT_ACTION_SKIPPED, XT_ACTION_PASSED, XT_ACTION_FAILED } xnupost_test_action_t;

typedef kern_return_t (*test_function)(void);
struct xnupost_test {
	uint16_t xt_config;
	uint16_t xt_test_num;
	kern_return_t xt_retval;
	kern_return_t xt_expected_retval;
	uint64_t xt_begin_time;
	uint64_t xt_end_time;
	xnupost_test_action_t xt_test_actions;
	test_function xt_func;
	const char * xt_name;
};

typedef kern_return_t xt_panic_return_t;
#define XT_PANIC_UNRELATED  0x8  /* not related. continue panic */
#define XT_RET_W_FAIL       0x9  /* report FAILURE and return from panic */
#define XT_RET_W_SUCCESS    0xA  /* report SUCCESS and return from panic */
#define XT_PANIC_W_FAIL     0xB  /* report FAILURE and continue to panic */
#define XT_PANIC_W_SUCCESS  0xC  /* report SUCCESS and continue to panic */

typedef xt_panic_return_t (*xt_panic_widget_func)(const char * panicstr, void * context, void ** outval);
struct xnupost_panic_widget {
	void * xtp_context_p;
	void ** xtp_outval_p;
	const char * xtp_func_name;
	xt_panic_widget_func xtp_func;
};

/* for internal use only. Use T_REGISTER_* macros */
extern xt_panic_return_t _xt_generic_assert_check(const char * s, void * str_to_match, void ** outval);
kern_return_t xnupost_register_panic_widget(xt_panic_widget_func funcp, const char * funcname, void * context, void ** outval);

#define T_REGISTER_PANIC_WIDGET(func, ctx, outval) xnupost_register_panic_widget((func), #func, (ctx), (outval))
#define T_REGISTER_ASSERT_CHECK(assert_str, retval) \
	T_REGISTER_PANIC_WIDGET(_xt_generic_assert_check, (void *)__DECONST(char *, assert_str), retval)

typedef struct xnupost_test xnupost_test_data_t;
typedef struct xnupost_test * xnupost_test_t;

extern struct xnupost_test kernel_post_tests[];
extern uint32_t kernel_post_tests_count;
extern uint32_t total_post_tests_count;

#define XNUPOST_TEST_CONFIG_BASIC(func)                   \
	{                                                 \
	        .xt_config = XT_CONFIG_RUN,               \
	        .xt_test_num = 0,                         \
	        .xt_retval = -1,                          \
	        .xt_expected_retval = T_STATE_PASS,       \
	        .xt_begin_time = 0,                       \
	        .xt_end_time = 0,                         \
	        .xt_test_actions = 0,                     \
	        .xt_func = (func),                        \
	        .xt_name = "xnu."#func                    \
	}

#define XNUPOST_TEST_CONFIG_TEST_PANIC(func)                       \
	{                                                          \
	        .xt_config = XT_CONFIG_EXPECT_PANIC,               \
	        .xt_test_num = 0,                                  \
	        .xt_retval = -1,                                   \
	        .xt_expected_retval = T_STATE_PASS,                \
	        .xt_begin_time = 0,                                \
	        .xt_end_time = 0,                                  \
	        .xt_test_actions = 0,                              \
	        .xt_func = (func),                                 \
	        .xt_name = "xnu."#func                             \
	}

void xnupost_init(void);
/*
 * Parse boot-args specific to POST testing and setup enabled/disabled settings
 * returns: KERN_SUCCESS - if testing is enabled.
 */
kern_return_t xnupost_parse_config(void);
kern_return_t xnupost_run_tests(xnupost_test_t test_list, uint32_t test_count);
kern_return_t xnupost_list_tests(xnupost_test_t test_list, uint32_t test_count);
kern_return_t xnupost_reset_tests(xnupost_test_t test_list, uint32_t test_count);

int xnupost_export_testdata(void * outp, size_t size, uint32_t * lenp);
uint32_t xnupost_get_estimated_testdata_size(void);

kern_return_t kernel_do_post(void);
kern_return_t xnupost_process_kdb_stop(const char * panic_s);
int xnupost_reset_all_tests(void);

kern_return_t kernel_list_tests(void);
int bsd_do_post(void);
int bsd_list_tests(void);

#endif /* _TESTS_XNUPOST_H */
