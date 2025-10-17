/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#include <stdbool.h>

#include <sys/errno.h>

#include <sys/kern_debug.h>

/* Syscall entry points */
int __debug_syscall_reject_config(uint64_t packed_selectors1, uint64_t packed_selectors2, uint64_t flags);

static bool supported = true;

typedef uint64_t packed_selector_t;

int
debug_syscall_reject_config(const syscall_rejection_selector_t *selectors, size_t len, uint64_t flags)
{
	_Static_assert(sizeof(syscall_rejection_selector_t) == 1, "selector size is not 1 byte");

	if (!supported) {
		/* Gracefully ignored if unsupported (e.g. if compiled out of RELEASE). */
		return 0;
	}

	if (len > (2 * 8 * sizeof(packed_selector_t)) / SYSCALL_REJECTION_SELECTOR_BITS) {
		/* selectors are packed one per 7 bits into two uint64_ts */
		errno = E2BIG;
		return -1;
	}

	/*
	 * The masks to apply are passed to the kernel as packed selectors,
	 * which are just however many of the selector data type fit into one
	 * (or more) fields of the natural word size (i.e. a register). This
	 * avoids copying from user space.
	 *
	 * More specifically, at the time of this writing, a selector is 1
	 * byte wide, and there is only one uint64_t argument
	 * (args->packed_selectors), so up to 8 selectors can be specified,
	 * which are then stuffed into the 64 bits of the argument. If less
	 * than 8 masks are requested to be applied, the remaining selectors
	 * will just be left as 0, which naturally resolves as the "empty" or
	 * "NULL" mask that changes nothing.
	 *
	 * This libsyscall wrapper provides a more convenient interface where
	 * an array (up to 8 elements long) and its length are passed in,
	 * which the wrapper then packs into packed_selectors of the actual
	 * system call.
	 */

	uint64_t packed_selectors[2] = { 0 };
	int shift = 0;

#define s_left_shift(x, n) ((n) < 0 ? ((x) >> -(n)) : ((x) << (n)))

	for (int i = 0; i < len; i++, shift += SYSCALL_REJECTION_SELECTOR_BITS) {
		int const second_shift = shift - 64;

		if (shift < 8 * sizeof(packed_selector_t)) {
			packed_selectors[0] |= ((uint64_t)(selectors[i]) & SYSCALL_REJECTION_SELECTOR_MASK) << shift;
		}
		if (second_shift > -SYSCALL_REJECTION_SELECTOR_BITS) {
			packed_selectors[1] |= s_left_shift((uint64_t)(selectors[i] & SYSCALL_REJECTION_SELECTOR_MASK), second_shift);
		}
	}

	int ret = __debug_syscall_reject_config(packed_selectors[0], packed_selectors[1], flags);

	if (ret == -1 && errno == ENOTSUP) {
		errno = 0;
		supported = false;
		return 0;
	}

	return ret;
}

/* Compatibility to old system call. */
int
debug_syscall_reject(const syscall_rejection_selector_t *selectors, size_t len)
{
	return debug_syscall_reject_config(selectors, len, SYSCALL_REJECTION_FLAGS_DEFAULT);
}
