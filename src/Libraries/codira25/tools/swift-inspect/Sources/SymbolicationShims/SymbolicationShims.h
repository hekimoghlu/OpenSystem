/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#if defined(__APPLE__)

#include <mach/vm_param.h>
#include <stdint.h>
#include <ptrauth.h>

struct CSTypeRef {
  uintptr_t a, b;
};

struct Range {
  uintptr_t location, length;
};

static inline uintptr_t GetPtrauthMask(void) {
#if __has_feature(ptrauth_calls)
  return (uintptr_t)ptrauth_strip((void*)0x0007ffffffffffff, 0);
#elif __arm64__ && __LP64__
  // Mask all bits above the top of MACH_VM_MAX_ADDRESS, which will
  // match the above ptrauth_strip.
  return (uintptr_t)~0ull >> __builtin_clzll(MACH_VM_MAX_ADDRESS);
#else
  return (uintptr_t)~0ull;
#endif
}

#endif
