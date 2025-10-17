/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#pragma once

#include <stddef.h>
#include <stdint.h>

/*
 * Implement fast stack unwinding for stack frames with frame pointers. Stores at most num_entries
 * return addresses to buffer buf. Returns the number of available return addresses, which may be
 * greater than num_entries.
 *
 * This function makes no guarantees about its behavior on encountering a frame built without frame
 * pointers, except that it should not crash or enter an infinite loop, and that any frames prior to
 * the frame built without frame pointers should be correct.
 *
 * This function is only meant to be used with memory safety tools such as sanitizers which need to
 * take stack traces efficiently. Normal applications should use APIs such as libunwindstack or
 * _Unwind_Backtrace.
 */
extern "C" size_t android_unsafe_frame_pointer_chase(uintptr_t* _Nonnull buf, size_t num_entries);
