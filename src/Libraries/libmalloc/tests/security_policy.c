/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

#include <darwintest.h>

#include <malloc_private.h>

T_DECL(security_policy_default,
		"Ensure that internal security is not enabled by default",
		T_META_TAG_NO_ALLOCATOR_OVERRIDE)
{
	T_ASSERT_FALSE(malloc_allows_internal_security_4test(),
			"Internal security should be disabled by default");
}

T_DECL(security_policy_envvar,
		"Ensure that internal security can be enabled via environment",
		T_META_TAG_NO_ALLOCATOR_OVERRIDE,
		T_META_ENVVAR("MallocAllowInternalSecurity=1"))
{
	T_ASSERT_TRUE(malloc_allows_internal_security_4test(),
			"Internal security should be enabled by the environment");
}
