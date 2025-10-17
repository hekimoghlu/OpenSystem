/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

//===--- SILAllocated.h - Defines the SILAllocated class --------*- C++ -*-===//
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

#ifndef LANGUAGE_SIL_SILALLOCATED_H
#define LANGUAGE_SIL_SILALLOCATED_H

#include "language/Basic/Toolchain.h"
#include "language/Basic/Compiler.h"
#include "toolchain/Support/ErrorHandling.h"
#include <stddef.h>

namespace language {
  class SILModule;

/// SILAllocated - This class enforces that derived classes are allocated out of
/// the SILModule bump pointer allocator.
template <typename DERIVED>
class SILAllocated {
public:
  /// Disable non-placement new.
  void *operator new(size_t) = delete;
  void *operator new[](size_t) = delete;

  /// Disable non-placement delete.
  void operator delete(void *) = delete;
  void operator delete[](void *) = delete;

  /// Custom version of 'new' that uses the SILModule's BumpPtrAllocator with
  /// precise alignment knowledge.  This is templated on the allocator type so
  /// that this doesn't require including SILModule.h.
  template <typename ContextTy>
  void *operator new(size_t Bytes, const ContextTy &C,
                     size_t Alignment = alignof(DERIVED)) {
    return C.allocate(Bytes, Alignment);
  }
};

} // end language namespace

#endif
