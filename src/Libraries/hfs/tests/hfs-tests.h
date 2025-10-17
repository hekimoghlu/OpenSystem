/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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

//
//  hfs-tests.h
//  hfs
//
//  Created by Chris Suter on 8/11/15.
//
//

#ifndef hfs_tests_c
#define hfs_tests_c

#include <sys/cdefs.h>
#include <stdbool.h>

__BEGIN_DECLS

typedef struct test_ctx {
	int reserved;
} test_ctx_t;

typedef int (* test_fn_t)(test_ctx_t *);

typedef struct test {
    const char *name;
	test_fn_t test_fn;
    bool run_as_root;
    // Other attributes
} test_t;

#define TEST_DECL(test_name, ...)						\
	static int run_##test_name(test_ctx_t *);			\
    struct test test_##test_name = {					\
		.name = #test_name,								\
		.test_fn = run_##test_name,						\
		__VA_ARGS__										\
    };													\
	static __attribute__((constructor))					\
	void register_test_##test_name(void) {				\
		register_test(&test_##test_name);				\
	}

#ifndef TEST
#define TEST(x, ...)	TEST_DECL(x, ##__VA_ARGS__)
#endif

void register_test(test_t *test);

__END_DECLS

#endif /* hfs_tests_c */
