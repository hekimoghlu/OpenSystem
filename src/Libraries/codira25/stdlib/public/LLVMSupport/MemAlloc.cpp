/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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

//===- MemAlloc.cpp - Memory allocation functions -------------------------===//
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

#include "toolchain/Support/MemAlloc.h"

// These are out of line to have __cpp_aligned_new not affect ABI.

TOOLCHAIN_ATTRIBUTE_RETURNS_NONNULL TOOLCHAIN_ATTRIBUTE_RETURNS_NOALIAS void *
toolchain::allocate_buffer(size_t Size, size_t Alignment) {
  return ::operator new(Size
#ifdef __cpp_aligned_new
                        ,
                        std::align_val_t(Alignment)
#endif
  );
}

void toolchain::deallocate_buffer(void *Ptr, size_t Size, size_t Alignment) {
  ::operator delete(Ptr
#ifdef __cpp_sized_deallocation
                    ,
                    Size
#endif
#ifdef __cpp_aligned_new
                    ,
                    std::align_val_t(Alignment)
#endif
  );
}
