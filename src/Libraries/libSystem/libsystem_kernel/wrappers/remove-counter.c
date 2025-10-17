/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#include <sys/types.h>
#if defined(__arm__)
#include <arm/arch.h>
#endif

#if defined(__ppc64__) || defined(__i386__) || defined(__x86_64__)
static int64_t __remove_counter = 0;
#else
static int32_t __remove_counter = 0;
#endif

__uint64_t
__get_remove_counter(void)
{
#if defined(__arm__) && !defined(_ARM_ARCH_6)
	return __remove_counter;
#else
	return __sync_add_and_fetch(&__remove_counter, 0);
#endif
}

void
__inc_remove_counter(void)
{
#if defined(__arm__) && !defined(_ARM_ARCH_6)
	__remove_counter++;
#else
	__sync_add_and_fetch(&__remove_counter, 1);
#endif
}
