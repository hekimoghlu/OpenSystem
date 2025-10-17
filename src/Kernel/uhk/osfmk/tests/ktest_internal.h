/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#ifndef _TESTS_KTEST_INTERNAL_H
#define _TESTS_KTEST_INTERNAL_H

#include <tests/ktest.h>
#include <stdint.h>

#define KTEST_VERSION 1
#define KTEST_VERSION_STR T_TOSTRING(KTEST_VERSION)

#define KTEST_MAXLEN 1024
#define KTEST_MAXOUTLEN 4096
#define KTEST_MAXVARS 3

#define KTEST_NUM_TESTCASE_MODES 2
#define KTEST_NUM_TESTCASE_STATES 4
#define KTEST_NUM_TEST_STATES 4

extern unsigned int ktest_current_line;
extern const char * ktest_current_file;
extern const char * ktest_current_func;
extern uint64_t ktest_current_time;

extern const char * ktest_test_name;

extern char ktest_current_msg[KTEST_MAXLEN];
extern char ktest_current_expr[KTEST_MAXOUTLEN];
extern char ktest_current_var_names[KTEST_MAXVARS][KTEST_MAXLEN];
extern char ktest_current_var_values[KTEST_MAXVARS][KTEST_MAXLEN];
extern unsigned int ktest_expression_index;
extern unsigned int ktest_current_var_index;
extern unsigned int ktest_test_index;
extern unsigned int ktest_passcount;
extern unsigned int ktest_failcount;
extern unsigned int ktest_xpasscount;
extern unsigned int ktest_xfailcount;
extern int ktest_expectfail;

extern int ktest_testcase_result;
extern int ktest_test_result;
extern int ktest_testcase_mode;

extern ktest_temp ktest_temp1, ktest_temp2, ktest_temp3;

extern char ktest_output_buf[KTEST_MAXLEN];

extern int ktest_test_result_statetab[KTEST_NUM_TEST_STATES]
[KTEST_NUM_TESTCASE_STATES]
[KTEST_NUM_TESTCASE_MODES];

extern const char * ktest_testcase_result_tokens[KTEST_NUM_TESTCASE_MODES]
[KTEST_NUM_TESTCASE_STATES];


void ktest_emit_start(void);
void ktest_emit_finish(void);
void ktest_emit_testbegin(const char * test_name);
void ktest_emit_testskip(const char * skip_msg, va_list args);
void ktest_emit_testend(void);
void ktest_emit_log(const char * log_msg, va_list args);
void ktest_emit_perfdata(const char * metric, const char * unit, double value, const char * desc);
void ktest_emit_testcase(void);

#endif /* _TESTS_KTEST_INTERNAL_H */
