/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
#include <stdint.h>
#include <kern/misc_protos.h>
#include <tests/ktest_internal.h>

unsigned int ktest_current_line = 0;
const char * ktest_current_file = NULL;
const char * ktest_current_func = NULL;
uint64_t ktest_current_time = 0;

const char * ktest_test_name = "";

char ktest_current_msg[KTEST_MAXLEN] = "";
char ktest_current_expr[KTEST_MAXOUTLEN] = "";
char ktest_current_var_names[KTEST_MAXVARS][KTEST_MAXLEN] = { "", "", ""  };
char ktest_current_var_values[KTEST_MAXVARS][KTEST_MAXLEN] = { "", "", "" };
unsigned int ktest_expression_index = 0;
unsigned int ktest_current_var_index = 0;
unsigned int ktest_test_index = 0;
unsigned int ktest_passcount = 0;
unsigned int ktest_failcount = 0;
unsigned int ktest_xpasscount = 0;
unsigned int ktest_xfailcount = 0;
int ktest_expectfail = 0;
int ktest_quiet = 0;

int ktest_testcase_result = T_RESULT_FAIL;
int ktest_test_result = T_STATE_UNRESOLVED;
int ktest_testcase_mode = T_MAIN;

ktest_temp ktest_temp1, ktest_temp2, ktest_temp3;

char ktest_output_buf[KTEST_MAXLEN] = "";

int
    ktest_test_result_statetab[KTEST_NUM_TEST_STATES]
[KTEST_NUM_TESTCASE_STATES]
[KTEST_NUM_TESTCASE_MODES] = {
	[T_STATE_UNRESOLVED][T_RESULT_PASS][T_MAIN] = T_STATE_PASS,
	[T_STATE_UNRESOLVED][T_RESULT_FAIL][T_MAIN] = T_STATE_FAIL,
	[T_STATE_UNRESOLVED][T_RESULT_UXPASS][T_MAIN] = T_STATE_FAIL,
	[T_STATE_UNRESOLVED][T_RESULT_XFAIL][T_MAIN] = T_STATE_PASS,

	[T_STATE_PASS][T_RESULT_PASS][T_MAIN] = T_STATE_PASS,
	[T_STATE_PASS][T_RESULT_FAIL][T_MAIN] = T_STATE_FAIL,
	[T_STATE_PASS][T_RESULT_UXPASS][T_MAIN] = T_STATE_FAIL,
	[T_STATE_PASS][T_RESULT_XFAIL][T_MAIN] = T_STATE_PASS,

	[T_STATE_FAIL][T_RESULT_PASS][T_MAIN] = T_STATE_FAIL,
	[T_STATE_FAIL][T_RESULT_FAIL][T_MAIN] = T_STATE_FAIL,
	[T_STATE_FAIL][T_RESULT_UXPASS][T_MAIN] = T_STATE_FAIL,
	[T_STATE_FAIL][T_RESULT_XFAIL][T_MAIN] = T_STATE_FAIL,

	[T_STATE_SETUPFAIL][T_RESULT_PASS][T_MAIN] = T_STATE_SETUPFAIL,
	[T_STATE_SETUPFAIL][T_RESULT_FAIL][T_MAIN] = T_STATE_SETUPFAIL,
	[T_STATE_SETUPFAIL][T_RESULT_UXPASS][T_MAIN] = T_STATE_SETUPFAIL,
	[T_STATE_SETUPFAIL][T_RESULT_XFAIL][T_MAIN] = T_STATE_SETUPFAIL,

	[T_STATE_UNRESOLVED][T_RESULT_PASS][T_SETUP] = T_STATE_UNRESOLVED,
	[T_STATE_UNRESOLVED][T_RESULT_FAIL][T_SETUP] = T_STATE_SETUPFAIL,
	[T_STATE_UNRESOLVED][T_RESULT_UXPASS][T_SETUP] = T_STATE_SETUPFAIL,
	[T_STATE_UNRESOLVED][T_RESULT_XFAIL][T_SETUP] = T_STATE_UNRESOLVED,

	[T_STATE_PASS][T_RESULT_PASS][T_SETUP] = T_STATE_PASS,
	[T_STATE_PASS][T_RESULT_FAIL][T_SETUP] = T_STATE_SETUPFAIL,
	[T_STATE_PASS][T_RESULT_UXPASS][T_SETUP] = T_STATE_SETUPFAIL,
	[T_STATE_PASS][T_RESULT_XFAIL][T_SETUP] = T_STATE_PASS,

	[T_STATE_FAIL][T_RESULT_PASS][T_SETUP] = T_STATE_FAIL,
	[T_STATE_FAIL][T_RESULT_FAIL][T_SETUP] = T_STATE_FAIL,
	[T_STATE_FAIL][T_RESULT_UXPASS][T_SETUP] = T_STATE_FAIL,
	[T_STATE_FAIL][T_RESULT_XFAIL][T_SETUP] = T_STATE_FAIL,

	[T_STATE_SETUPFAIL][T_RESULT_PASS][T_SETUP] = T_STATE_SETUPFAIL,
	[T_STATE_SETUPFAIL][T_RESULT_FAIL][T_SETUP] = T_STATE_SETUPFAIL,
	[T_STATE_SETUPFAIL][T_RESULT_UXPASS][T_SETUP] = T_STATE_SETUPFAIL,
	[T_STATE_SETUPFAIL][T_RESULT_XFAIL][T_SETUP] = T_STATE_SETUPFAIL,
};

const char * ktest_testcase_result_tokens[KTEST_NUM_TESTCASE_MODES]
[KTEST_NUM_TESTCASE_STATES] = {
	[T_MAIN][T_RESULT_PASS] = "PASS",
	[T_MAIN][T_RESULT_FAIL] = "FAIL",
	[T_MAIN][T_RESULT_UXPASS] = "UXPASS",
	[T_MAIN][T_RESULT_XFAIL] = "XFAIL",
	[T_SETUP][T_RESULT_PASS] = "SETUP_PASS",
	[T_SETUP][T_RESULT_FAIL] = "SETUP_FAIL",
	[T_SETUP][T_RESULT_UXPASS] = "SETUP_UXPASS",
	[T_SETUP][T_RESULT_XFAIL] = "SETUP_XFAIL",
};
