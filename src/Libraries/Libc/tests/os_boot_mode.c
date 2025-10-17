/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

#include <os/boot_mode_private.h>
#include <TargetConditionals.h>

#include <darwintest.h>

#if TARGET_OS_OSX
T_DECL(os_boot_mode_basic, "Can't know our exact boot mode, but it should be fetchable")
{
	const char *boot_mode = "??????";
	bool result = os_boot_mode_query(&boot_mode);
	if (result && !boot_mode) {
		boot_mode = "no-particular-mode";
	}
	T_ASSERT_TRUE(result, "os_boot_mode_query() success (%s)", boot_mode);
	T_ASSERT_NE_STR(boot_mode, "??????", "we actually set the result");
}
#endif
