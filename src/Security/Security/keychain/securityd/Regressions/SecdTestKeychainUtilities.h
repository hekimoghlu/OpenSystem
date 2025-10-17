/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#ifndef _SECDTESTKEYCHAINUTILITIES_
#define _SECDTESTKEYCHAINUTILITIES_

#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>

#define kSecdTestSetupTestCount 1
void secd_test_setup_temp_keychain(const char* test_prefix, dispatch_block_t do_before_reset);
bool secd_test_teardown_delete_temp_keychain(const char* test_prefix);

extern CFStringRef kTestView1;
extern CFStringRef kTestView2;

void secd_test_setup_testviews(void);
void secd_test_clear_testviews(void);

#endif
