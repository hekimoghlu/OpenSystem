/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#ifndef __DEBUG_H
#define __DEBUG_H

// set to one to debug malloc itself
#define DEBUG_MALLOC 0
// set to one to debug malloc client
#define DEBUG_CLIENT 0
#define DEBUG_MADVISE 0

#if DEBUG_MALLOC
#   warning DEBUG_MALLOC ENABLED
#   undef MALLOC_INLINE
#   undef MALLOC_UNUSED
#   undef MALLOC_ALWAYS_INLINE
#   undef CHECK_MAGAZINE_PTR_LOCKED

#   define MALLOC_INLINE
#   define MALLOC_UNUSED
#   define MALLOC_ALWAYS_INLINE
#   define CHECK_MAGAZINE_PTR_LOCKED(szone, mag_ptr, fun)				\
	do {										\
	    if (TRY_LOCK(mag_ptr->magazine_lock)) {					\
		malloc_report(ASL_LEVEL_ERR, "*** magazine_lock was not set %p in %s\n",		\
		mag_ptr->magazine_lock, fun);						\
	    }										\
	} while (0)
#endif // DEBUG_MALLOC

#if DEBUG_MALLOC || DEBUG_CLIENT
#	define CHECK(szone, fun) \
	if ((szone)->debug_flags & CHECK_REGIONS) { \
		szone_check_all(szone, fun) \
	}
#else // DEBUG_MALLOC || DEBUG_CLIENT
#	define CHECK(szone, fun) \
	do {} while (0)
#endif // DEBUG_MALLOC || DEBUG_CLIENT

#endif // __DEBUG_H
