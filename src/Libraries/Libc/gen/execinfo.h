/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#ifndef _EXECINFO_H_
#define _EXECINFO_H_ 1

#include <sys/cdefs.h>
#include <_bounds.h>
#include <Availability.h>
#include <os/base.h>
#include <os/availability.h>
#include <stddef.h>
#include <stdint.h>
#include <uuid/uuid.h>

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS

int backtrace(void **_LIBC_COUNT(__size), int __size) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);

API_AVAILABLE(macosx(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_EXPORT
int backtrace_from_fp(void *startfp, void **_LIBC_COUNT(size) array, int size);

char *_LIBC_CSTR *_LIBC_COUNT_OR_NULL(__size) backtrace_symbols(void* const* _LIBC_COUNT(__size), int __size) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);
void backtrace_symbols_fd(void* const* _LIBC_COUNT(__size),int __size,int) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);

struct image_offset {
	/*
	 * The UUID of the image.
	 */
	uuid_t uuid;

	/*
	 * The offset is relative to the __TEXT section of the image.
	 */
	uint32_t offset;
};

API_AVAILABLE(macosx(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_EXPORT
void backtrace_image_offsets(void* const* _LIBC_COUNT(size) array,
		struct image_offset *image_offsets, int size);

/*!
 * @function backtrace_async
 * Extracts the function return addresses of the current call stack. While
 * backtrace() will only follow the OS call stack, backtrace_async() will
 * prefer the unwind the Swift concurrency continuation stack if invoked
 * from within an async context. In a non-async context this function is
 * strictly equivalent to backtrace().
 *
 * @param array
 * The array of pointers to fill with the return addresses.
 *
 * @param length
 * The maximum number of pointers to write.
 *
 * @param task_id
 * Can be NULL. If non-NULL, the uint32_t pointed to by `task_id` is set to
 * a non-zero value that for the current process uniquely identifies the async
 * task currently running. If called from a non-async context, the value is
 * set to 0 and `array` contains the same values backtrace() would return.
 *
 * Note that the continuation addresses provided by backtrace_async()
 * have an offset of 1 added to them.  Most symbolication engines will
 * substract 1 from the call stack return addresses in order to symbolicate
 * the call site rather than the return location.  With a Swift async
 * continuation, substracting 1 from its address would result in an address
 * in a different function.  This offset allows the returned addresses to be
 * handled correctly by most existing symbolication engines.
 *
 * @result
 * The number of pointers actually written.
 */
API_AVAILABLE(macosx(12.0), ios(15.0), tvos(15.0), watchos(8.0))
size_t backtrace_async(void** _LIBC_COUNT(length) array, size_t length, uint32_t *task_id);

__END_DECLS

#endif /* !_EXECINFO_H_ */
