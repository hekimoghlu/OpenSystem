/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#include <mach/mach_time.h>

extern kern_return_t
mach_timebase_info_trap(mach_timebase_info_t info);

kern_return_t
mach_timebase_info(mach_timebase_info_t info)
{
	static mach_timebase_info_data_t cached_info;

	/*
	 * This is racy, but because it is safe to initialize twice we avoid a
	 * barrier in the fast path by risking double initialization.
	 */
	if (cached_info.numer == 0 || cached_info.denom == 0) {
		kern_return_t kr = mach_timebase_info_trap(&cached_info);
		if (kr != KERN_SUCCESS) {
			return kr;
		}
	}

	info->numer = cached_info.numer;
	info->denom = cached_info.denom;

	return KERN_SUCCESS;
}
