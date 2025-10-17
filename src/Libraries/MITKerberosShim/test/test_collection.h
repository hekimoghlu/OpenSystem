/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
 * 1) Create an test abstraction layer for integration into other systems.
 * 2) Create a simplified way to collect and report test results.
 * 3) Low impact in the test source.
 *
 *
 * Convenience functions:
 * ----------------------
 * test_collection_t *tests_init_and_start(const char*);
 * int tests_stop_and_free(test_collection_t*);
 * time_t tests_start_timer(test_collection_t*);
 * time_t tests_stop_timer(test_collection_t*);
 *
 * Test status functions:
 * ----------------------
 * void test_passed(test_collection_t*, const char*);
 * void test_failed(test_collection_t*, const char*, const char*, ...)
 * int test_evaluate(test_collection_t*, const char*, int, const char*, ...)
 *
 * Setting library options:
 * ------------------------
 * uint32_t tests_set_flags(test_collection_t*, uint32_t);
 * uint32_t tests_unset_flags(test_collection_t*, uint32_t);
 *
 * Other important functions:
 * --------------------------
 * size_t tests_set_total_count_hint(test_collection_t*, size_t);
 * int tests_return_value(const test_collection_t*);
 * double tests_duration(test_collection_t*);
 *
 *
 */

#include <inttypes.h>
#include <stdarg.h>
#include <time.h>

#if !defined(_TEST_COLLECTION_H_)
#define _TEST_COLLECTION_H_

/* Possible test statuses.  */
enum test_collection_return_values {
	TC_TESTS_PASSED = 0,
	TC_TESTS_FAILED
};

/* Maximum string length for name. */
#define TC_NAME_MAX_LENGTH 512

/* Available flags. */
#define TC_FLAG_NONE                0u
#define TC_FLAG_EXIT_ON_FAILURE (1u<<1)
#define TC_FLAG_SUMMARY_ON_STOP (1u<<2)

#define TC_FLAG_DEFAULTS (TC_FLAG_SUMMARY_ON_STOP)

/* Structure representing a collections of tests. */
typedef struct _struct_test_collection_t {
	char *name;
	size_t failed_count;
	size_t passed_count;
	size_t total_count_hint;
	time_t start_time;
	time_t stop_time;
	uint32_t flags;
} test_collection_t;

test_collection_t *tests_init_and_start(const char*);
int tests_stop_and_free(test_collection_t*);

void test_passed(test_collection_t*, const char*);
void test_failed(test_collection_t*, const char*, const char*, ...)
	__attribute__((format(printf, 2, 4)));
int test_evaluate(test_collection_t*, const char*, int);
int vtest_evaluate(test_collection_t*, const char*, int, const char*, ...)
	__attribute__((format(printf, 4, 5)));


test_collection_t *test_collection_init(const char*);
void test_collection_free(test_collection_t*);

int tests_return_value(const test_collection_t*);

time_t tests_start_timer(test_collection_t*);
time_t tests_stop_timer(test_collection_t*);
time_t tests_get_start_time(test_collection_t*);
time_t tests_set_start_time(test_collection_t*, time_t);
time_t tests_get_stop_time(test_collection_t*);
time_t tests_set_stop_time(test_collection_t*, time_t);
double tests_duration(test_collection_t*);

uint32_t tests_get_flags(const test_collection_t*);
uint32_t tests_set_flags(test_collection_t*, uint32_t);
uint32_t tests_unset_flags(test_collection_t*, uint32_t);

size_t tests_get_total_count_hint(const test_collection_t*);
size_t tests_set_total_count_hint(test_collection_t*, size_t);

char *tests_get_name(const test_collection_t*);
char *tests_set_name(test_collection_t*, const char*);

#endif /* _TEST_COLLECTION_H_ */

