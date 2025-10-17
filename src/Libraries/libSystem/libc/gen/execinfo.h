/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#include <Availability.h>
#include <os/base.h>
#include <os/availability.h>
#include <stdint.h>
#include <uuid/uuid.h>

__BEGIN_DECLS

int backtrace(void**,int) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);

API_AVAILABLE(macosx(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_EXPORT
int backtrace_from_fp(void *startfp, void **array, int size);

char** backtrace_symbols(void* const*,int) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);
void backtrace_symbols_fd(void* const*,int,int) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);

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
void backtrace_image_offsets(void* const* array,
		struct image_offset *image_offsets, int size);

__END_DECLS

#endif /* !_EXECINFO_H_ */
