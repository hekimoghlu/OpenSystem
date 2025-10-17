/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#include "internal.h"
#include <setjmp.h>

bool
malloc_tracing_enabled = false;

malloc_zero_policy_t malloc_zero_policy = MALLOC_ZERO_POLICY_DEFAULT;
unsigned malloc_zero_on_free_sample_period = 0;
#if CONFIG_CLUSTER_AWARE
unsigned int ncpuclusters = 1;
#endif

void
malloc_zone_check_fail(const char *msg, const char *fmt, ...)
{
	__builtin_trap();
}

void
malloc_error_break(void)
{
	__builtin_trap();
}

jmp_buf *zone_error_expected_jmp;

void
malloc_zone_error(uint32_t flags, bool is_corruption, const char *fmt, ...)
{
	if (!zone_error_expected_jmp || !is_corruption) {
		__builtin_trap();
	}

	longjmp(*zone_error_expected_jmp, 1);
}

void
find_zone_and_free(void *ptr, bool known_non_default)
{
	__builtin_trap();
}
