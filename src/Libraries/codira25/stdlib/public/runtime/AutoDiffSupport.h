/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

//===--- AutoDiffSupport.h ------------------------------------*- C++ -*---===//
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

#ifndef LANGUAGE_RUNTIME_AUTODIFF_SUPPORT_H
#define LANGUAGE_RUNTIME_AUTODIFF_SUPPORT_H

#include "language/Runtime/HeapObject.h"
#include "language/ABI/Metadata.h"
#include "language/Runtime/Config.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "toolchain/Support/Allocator.h"

namespace language {
/// A data structure responsible for efficiently allocating closure contexts for
/// linear maps such as pullbacks, including recursive branching trace enum
/// case payloads.
class AutoDiffLinearMapContext : public HeapObject {
  /// A simple wrapper around a context object allocated by the
  /// `AutoDiffLinearMapContext` type. This type knows all the "physical"
  /// properties and behavior of the allocated context object by way of
  /// storing the allocated type's `TypeMetadata`. It uses this information
  /// to ensure that the allocated context object is destroyed/deinitialized
  /// properly, upon its own destruction.
  class [[nodiscard]] AllocatedContextObjectRecord final {
    const Metadata *contextObjectMetadata;
    OpaqueValue *contextObjectPtr;

  public:
    AllocatedContextObjectRecord(const Metadata *contextObjectMetadata,
                                 OpaqueValue *contextObjectPtr)
        : contextObjectMetadata(contextObjectMetadata),
          contextObjectPtr(contextObjectPtr) {}

    AllocatedContextObjectRecord(const Metadata *contextObjectMetadata,
                                 void *contextObjectPtr)
        : AllocatedContextObjectRecord(
              contextObjectMetadata,
              static_cast<OpaqueValue *>(contextObjectPtr)) {}

    ~AllocatedContextObjectRecord() {
      if (contextObjectMetadata != nullptr && contextObjectPtr != nullptr) {
        contextObjectMetadata->vw_destroy(contextObjectPtr);
      }
    }

    AllocatedContextObjectRecord(const AllocatedContextObjectRecord &) = delete;

    AllocatedContextObjectRecord(
        AllocatedContextObjectRecord &&other) noexcept {
      this->contextObjectMetadata = other.contextObjectMetadata;
      this->contextObjectPtr = other.contextObjectPtr;
      other.contextObjectMetadata = nullptr;
      other.contextObjectPtr = nullptr;
    }

    size_t size() const { return contextObjectMetadata->vw_size(); }

    size_t align() const { return contextObjectMetadata->vw_alignment(); }
  };

private:
  /// The underlying allocator.
  // TODO: Use a custom allocator so that the initial slab can be
  // tail-allocated.
  toolchain::BumpPtrAllocator allocator;

  /// Storage for `AllocatedContextObjectRecord`s, corresponding to the
  /// subcontext allocations performed by the type.
  toolchain::SmallVector<AllocatedContextObjectRecord, 4> allocatedContextObjects;

public:
  /// DEPRECATED - Use overloaded constructor taking a `const Metadata *`
  /// parameter instead. This constructor might be removed as it leads to memory
  /// leaks.
  AutoDiffLinearMapContext();

  AutoDiffLinearMapContext(const Metadata *topLevelLinearMapContextMetadata);

  /// Returns the address of the tail-allocated top-level subcontext.
  void *projectTopLevelSubcontext() const;

  /// Allocates memory for a new subcontext.
  ///
  /// DEPRECATED - Use `allocateSubcontext` instead. This
  /// method might be removed as it leads to memory leaks.
  void *allocate(size_t size);

  /// Allocates memory for a new subcontext.
  void *allocateSubcontext(const Metadata *contextObjectMetadata);
};

/// Creates a linear map context with a tail-allocated top-level subcontext.
///
/// DEPRECATED - Use `language_autoDiffCreateLinearMapContextWithType` instead.
/// This builtin might be removed as it leads to memory leaks.
LANGUAGE_RUNTIME_EXPORT LANGUAGE_CC(language)
AutoDiffLinearMapContext *language_autoDiffCreateLinearMapContext(
    size_t topLevelSubcontextSize);

/// Returns the address of the tail-allocated top-level subcontext.
LANGUAGE_RUNTIME_EXPORT LANGUAGE_CC(language)
void *language_autoDiffProjectTopLevelSubcontext(AutoDiffLinearMapContext *);

/// Allocates memory for a new subcontext.
///
/// DEPRECATED - Use `language_autoDiffAllocateSubcontextWithType` instead. This
/// builtin might be removed as it leads to memory leaks.
LANGUAGE_RUNTIME_EXPORT LANGUAGE_CC(language)
void *language_autoDiffAllocateSubcontext(AutoDiffLinearMapContext *, size_t size);

/// Creates a linear map context with a tail-allocated top-level subcontext.
LANGUAGE_RUNTIME_EXPORT LANGUAGE_CC(language)
    AutoDiffLinearMapContext *language_autoDiffCreateLinearMapContextWithType(
        const Metadata *topLevelLinearMapContextMetadata);

/// Allocates memory for a new subcontext.
LANGUAGE_RUNTIME_EXPORT
    LANGUAGE_CC(language) void *language_autoDiffAllocateSubcontextWithType(
        AutoDiffLinearMapContext *,
        const Metadata *linearMapSubcontextMetadata);
} // namespace language
#endif /* LANGUAGE_RUNTIME_AUTODIFF_SUPPORT_H */
