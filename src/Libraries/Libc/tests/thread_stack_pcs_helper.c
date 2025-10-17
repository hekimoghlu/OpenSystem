/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
// this file contains a dyld stack, which interposes the use of these symbols in libc for testing

#include <mach-o/dyld-interposing.h>
#include <mach-o/dyld_priv.h>

static uint8_t test_dyld_stack[32768];

static const void* test_dyld_stack_top = &test_dyld_stack[32768];
static const void* test_dyld_stack_bottom = &test_dyld_stack[0];

static void test_dyld_stack_range(const void** stack_bottom, const void** stack_top)
{
	*stack_bottom = test_dyld_stack_bottom;
	*stack_top = test_dyld_stack_top;
}

DYLD_INTERPOSE(test_dyld_stack_range, _dyld_stack_range)
