/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#include "cc_internal.h"

#if CC_DIT_MAYBE_SUPPORTED

// Ignore "unreachable code" warnings when compiling against SDKs
// that don't support checking for DIT support at runtime.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code"

void
cc_disable_dit(volatile bool *dit_was_enabled)
{
	if (!CC_HAS_DIT()) {
		return;
	}

#if CC_BUILT_FOR_TESTING
	// DIT should be enabled.
	cc_try_abort_if(!cc_is_dit_enabled(), "DIT not enabled");
#endif

	// Disable DIT, if this was the frame that enabled it.
	if (*dit_was_enabled) {
		// Encoding of <msr dit, #0>.
		__asm__ __volatile__ (".long 0xd503405f");
		cc_assert(!cc_is_dit_enabled());
	}
}

#pragma clang diagnostic pop

#endif // CC_DIT_MAYBE_SUPPORTED

