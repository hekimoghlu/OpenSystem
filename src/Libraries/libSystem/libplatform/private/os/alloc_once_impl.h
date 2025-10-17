/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
#ifndef __OS_ALLOC_ONCE_IMPL__
#define __OS_ALLOC_ONCE_IMPL__

#ifndef __OS_ALLOC_INDIRECT__
#error "Please include <os/alloc_once_private.h> instead of this file directly."
#endif

#include <Availability.h>
#include <sys/types.h>
#include <os/base_private.h>
#include <os/once_private.h>

__BEGIN_DECLS

#define OS_ALLOC_SPI_VERSION 20120430

#define OS_ALLOC_ONCE_KEY_MAX 100

typedef os_once_t os_alloc_token_t;
struct _os_alloc_once_s {
	os_alloc_token_t once;
	void *ptr;
};

__OSX_AVAILABLE_STARTING(__MAC_10_9,__IPHONE_6_0)
extern struct _os_alloc_once_s _os_alloc_once_table[];

__OSX_AVAILABLE_STARTING(__MAC_10_9,__IPHONE_6_0)
OS_EXPORT OS_NONNULL1
void*
_os_alloc_once(struct _os_alloc_once_s *slot, size_t sz, os_function_t init);

/* 
 * The region allocated by os_alloc_once is 0-filled when initially
 * returned (or handed off to the initializer).
 */
OS_WARN_RESULT OS_NOTHROW OS_CONST
__header_always_inline void*
os_alloc_once(os_alloc_token_t token, size_t sz, os_function_t init)
{
	struct _os_alloc_once_s *slot = &_os_alloc_once_table[token];
	if (OS_EXPECT(slot->once, ~0l) != ~0l) {
		void *ptr = _os_alloc_once(slot, sz, init);
		OS_COMPILER_CAN_ASSUME(slot->once == ~0l);
		return ptr;
	}
	return slot->ptr;
}

__END_DECLS

#endif // __OS_ALLOC_ONCE_IMPL__
