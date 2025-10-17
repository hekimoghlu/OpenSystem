/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#define RO0 23
#define RO1 234
#define RW0 2345
#define RW1 23456
#define BSS0 0
#define BSS1 0

#define RW0_INCREMENT 12
#define RW1_INCREMENT 123
#define BSS0_INCREMENT 1234
#define BSS1_INCREMENT 12345

#define TEST_RESULT_BASE (RO0 + RO1 + RW0 + RW1 + BSS0 + BSS1 + RW0)
#define TEST_RESULT_INCREMENT \
  (RW0_INCREMENT + RW1_INCREMENT + BSS0_INCREMENT + BSS1_INCREMENT + RW0_INCREMENT)

typedef int (*loader_test_func_t)(void);
