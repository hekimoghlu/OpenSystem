/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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

//===--- AsyncLet.h - ABI structures for async let -00-----------*- C++ -*-===//
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
//
// Codira ABI describing task groups.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_ABI_TASK_ASYNC_LET_H
#define LANGUAGE_ABI_TASK_ASYNC_LET_H

#include "language/ABI/Task.h"
#include "language/ABI/HeapObject.h"
#include "language/Runtime/Concurrency.h"
#include "language/Runtime/Config.h"
#include "language/Basic/RelativePointer.h"
#include "language/Basic/STLExtras.h"

namespace language {

/// Represents an in-flight `async let`, i.e. the Task that is computing the
/// result of the async let, along with the awaited status and other metadata.
class alignas(Alignment_AsyncLet) AsyncLet {
public:
  // These constructors do not initialize the AsyncLet instance, and the
  // destructor does not destroy the AsyncLet instance; you must call
  // language_asyncLet_{start,end} yourself.
  constexpr AsyncLet()
    : PrivateData{} {}

  void *PrivateData[NumWords_AsyncLet];

  // TODO: we could offer a "was awaited on" check here

  /// Returns the child task that is associated with this async let.
  /// The tasks completion is used to fulfil the value represented by this async let.
  AsyncTask *getTask() const;
  
  // The compiler preallocates a large fixed space for the `async let`, with the
  // intent that most of it be used for the child task context. The next two
  // methods return the address and size of that space.
  
  /// Return a pointer to the unused space within the async let block.
  void *getPreallocatedSpace();
  
  /// Return the size of the unused space within the async let block.
  static size_t getSizeOfPreallocatedSpace();
  
  /// Was the task allocated out of the parent's allocator?
  bool didAllocateFromParentTask();
  
  /// Flag that the task was allocated from the parent's allocator.
  void setDidAllocateFromParentTask(bool value = true);
};

} // end namespace language

#endif // LANGUAGE_ABI_TASK_ASYNC_LET_H
