/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
//  codesigning_tests_shared.h
//  Security
//
//  Copyright 2021 Apple Inc. All rights reserved.
//

//
// BATS test token helpers
//
#define TEST_START(name) \
    do { \
        printf("==================================================\n"); \
        printf("[TEST] %s\n", name); \
        printf("==================================================\n"); \
    } while(0)

#define TEST_CASE(cond, name) \
    do { \
        printf("[BEGIN] %s\n", (name)); \
        if ((cond)) \
            printf("[PASS] %s\n", (name)); \
        else \
            printf("[FAIL] %s\n", (name)); \
    } while (0)

#define TEST_CASE_EXPR(cond) TEST_CASE(cond, #cond)

#define TEST_CASE_JUMP(cond, block, name) \
    do { \
        printf("[BEGIN] %s\n", (name)); \
        if ((cond)) \
            printf("[PASS] %s\n", (name)); \
        else  {\
            printf("[FAIL] %s\n", (name)); \
            goto block; \
        } \
    } while (0)

#define TEST_CASE_EXPR_JUMP(cond, block) TEST_CASE_JUMP(cond, block, #cond)

#define TEST_CASE_BLOCK(name, block) \
    do { \
        printf("[BEGIN] %s\n", (name)); \
        if (block()) \
            printf("[PASS] %s\n", (name)); \
        else \
            printf("[FAIL] %s\n", (name)); \
    } while (0)

#define TEST_BEGIN printf("[BEGIN] %s\n", __FUNCTION__);
#define TEST_PASS printf("[PASS] %s\n", __FUNCTION__);
#define TEST_FAIL printf("[FAIL] %s\n", __FUNCTION__);

#define TEST_RESULT(cond) \
    (cond) ? TEST_PASS : TEST_FAIL

//
// Common output helpers
//
#define INFO(fmt, ...)                                      \
({                                                          \
    NSLog(fmt, ##__VA_ARGS__);                              \
})
#define PASS(fmt, ...)                                                      \
({                                                                          \
    fprintf(stdout, "[PASS] %s " fmt "\n", __FUNCTION__, ##__VA_ARGS__);    \
})
#define FAIL(fmt, ...)                                                      \
({                                                                          \
    fprintf(stdout, "[FAIL] %s " fmt "\n", __FUNCTION__, ##__VA_ARGS__);    \
})
