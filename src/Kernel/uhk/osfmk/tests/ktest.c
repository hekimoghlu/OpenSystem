/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#include <tests/ktest.h>
#include <tests/ktest_internal.h>
#include <mach/mach_time.h>
#include <kern/misc_protos.h>

void
ktest_start(void)
{
	ktest_emit_start();
}

void
ktest_finish(void)
{
	ktest_emit_finish();
}

void
ktest_testbegin(const char * test_name)
{
	ktest_current_time = mach_absolute_time();
	ktest_test_name = test_name;
	ktest_emit_testbegin(test_name);
}

void
ktest_testend()
{
	ktest_current_time = mach_absolute_time();
	ktest_emit_testend();
	ktest_test_index++;
}

void
ktest_testskip(const char * msg, ...)
{
	va_list args;

	ktest_current_time = mach_absolute_time();

	va_start(args, msg);
	ktest_emit_testskip(msg, args);
	va_end(args);
}

void
ktest_log(const char * msg, ...)
{
	va_list args;

	ktest_current_time = mach_absolute_time();

	va_start(args, msg);
	ktest_emit_log(msg, args);
	va_end(args);
}

void
ktest_perf(const char * metric, const char * unit, double value, const char * desc)
{
	ktest_current_time = mach_absolute_time();
	ktest_emit_perfdata(metric, unit, value, desc);
}

void
ktest_testcase(int success)
{
	ktest_current_time = mach_absolute_time();

	if (success && !ktest_expectfail) {
		/* PASS */
		ktest_passcount++;
		ktest_testcase_result = T_RESULT_PASS;
	} else if (!success && !ktest_expectfail) {
		/* FAIL */
		ktest_failcount++;
		ktest_testcase_result = T_RESULT_FAIL;
	} else if (success && ktest_expectfail) {
		/* UXPASS */
		ktest_xpasscount++;
		ktest_testcase_result = T_RESULT_UXPASS;
	} else if (!success && ktest_expectfail) {
		/* XFAIL */
		ktest_xfailcount++;
		ktest_testcase_result = T_RESULT_XFAIL;
	}

	ktest_update_test_result_state();
	if (ktest_quiet == 0 ||
	    ktest_testcase_result == T_RESULT_FAIL ||
	    ktest_testcase_result == T_RESULT_UXPASS) {
		ktest_emit_testcase();
	}
	ktest_expression_index++;

	ktest_quiet = 0;
	ktest_expectfail = 0;
	ktest_output_buf[0] = '\0';
	ktest_current_msg[0] = '\0';
	ktest_current_expr[0] = '\0';
	for (int i = 0; i < KTEST_MAXVARS; i++) {
		ktest_current_var_names[i][0] = '\0';
		ktest_current_var_values[i][0] = '\0';
	}
	ktest_current_var_index = 0;
}

void
ktest_update_test_result_state(void)
{
	ktest_test_result = ktest_test_result_statetab[ktest_test_result]
	    [ktest_testcase_result]
	    [ktest_testcase_mode];
}

void
ktest_assertion_check(void)
{
	if (ktest_testcase_result == T_RESULT_FAIL || ktest_testcase_result == T_RESULT_UXPASS) {
		ktest_testend();
		panic("XNUPOST: Assertion failed : %s : at %s:%d", ktest_test_name, ktest_current_file, ktest_current_line);
	}
}
