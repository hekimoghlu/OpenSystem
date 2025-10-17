/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#pragma once

#define TEST_START(name) \
do { \
    printf("==================================================\n"); \
    printf("[TEST] %s\n", name); \
    printf("==================================================\n"); \
} while(0)

#define TEST_RESULT(name, cond) \
do { \
    if ((cond)) \
        printf("[PASS] %s\n", (name)); \
    else \
        printf("[FAIL] %s\n", (name)); \
} while (0)

#define TEST_REQUIRE(name, cond, errVar, errVal, label) \
do { \
    TEST_RESULT(name, cond); \
    if (!(cond)) { \
        (errVar) = (errVal); \
        goto label; \
    } \
} while (0)

#define TEST_CASE(name, cond) \
do { \
    printf("[BEGIN] %s\n", (name)); \
    TEST_RESULT(name, cond); \
} while (0)

#define TEST_LOG(fmt, ...) \
do { \
    printf("[LOG] %s.%d: " fmt "\n", \
            __FILE__, __LINE__, ##__VA_ARGS__); \
} while (0)
