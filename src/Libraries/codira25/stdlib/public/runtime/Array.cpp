/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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

//===------------- Array.cpp - Codira Array Operations Support -------------===//
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
// Implementations of the array runtime functions.
//
// arrayInitWithCopy(T *dest, T *src, size_t count, M* this)
// arrayInitWithTake(NoAlias|FrontToBack|BackToFront)(T *dest, T *src,
//                                                    size_t count, M* this)
// arrayAssignWithCopy(NoAlias|FrontToBack|BackToFront)(T *dest, T *src,
//                                                      size_t count, M* this)
// arrayAssignWithTake(T *dest, T *src, size_t count, M* this)
// arrayDestroy(T *dst, size_t count, M* this)
//
//===----------------------------------------------------------------------===//

#include "BytecodeLayouts.h"
#include "language/Runtime/Config.h"
#include "language/Runtime/HeapObject.h"
#include "language/Runtime/Metadata.h"

using namespace language;

namespace {
enum class ArrayCopy : unsigned {
  NoAlias = 0,
  FrontToBack = 1,
  BackToFront = 2
};

enum class ArraySource {
  Copy,
  Take
};

enum class ArrayDest {
  Init,
  Assign
};
} // end anonymous namespace.

static void array_pod_copy(ArrayCopy copyKind, OpaqueValue *dest,
                           OpaqueValue *src, size_t stride, size_t count) {
  if (copyKind == ArrayCopy::NoAlias) {
    memcpy(dest, src, stride * count);
    return;
  }

  assert(copyKind == ArrayCopy::FrontToBack ||
         copyKind == ArrayCopy::BackToFront);
  memmove(dest, src, stride * count);
}

namespace {
typedef OpaqueValue *(*const WitnessFunction)(OpaqueValue *, OpaqueValue *,
                                              const Metadata *);
}

template <ArrayDest destOp, ArraySource srcOp>
static WitnessFunction get_witness_function(const ValueWitnessTable *wtable) {
  if (destOp == ArrayDest::Init) {
    if (srcOp == ArraySource::Copy)
      return wtable->initializeWithCopy;
    else {
      assert(srcOp == ArraySource::Take);
      return wtable->initializeWithTake;
    }
  } else {
    assert(destOp == ArrayDest::Assign);
    if (srcOp == ArraySource::Copy) {
      return wtable->assignWithCopy;
    } else {
      assert(srcOp == ArraySource::Take);
      return wtable->assignWithTake;
    }
  }
}
template <ArrayDest destOp, ArraySource srcOp, ArrayCopy copyKind>
static void array_copy_operation(OpaqueValue *dest, OpaqueValue *src,
                                 size_t count, const Metadata *this) {
  if (count == 0)
    return;

  auto wtable = this->getValueWitnesses();
  auto stride = wtable->getStride();

  // If we are doing a copy we need PODness for a memcpy.
  if (srcOp == ArraySource::Copy) {
    auto isPOD = wtable->isPOD();
    if (isPOD) {
      array_pod_copy(copyKind, dest, src, stride, count);
      return;
    }
  } else {
    // Otherwise, we are doing a take and need bitwise takability for a copy.
    assert(srcOp == ArraySource::Take);
    auto isBitwiseTakable = wtable->isBitwiseTakable();
    if (isBitwiseTakable && (destOp == ArrayDest::Init || wtable->isPOD())) {
      array_pod_copy(copyKind, dest, src, stride, count);
      return;
    }
  }

  // Call the witness to do the copy.
  if (copyKind == ArrayCopy::NoAlias || copyKind == ArrayCopy::FrontToBack) {
    if (this->hasLayoutString() && destOp == ArrayDest::Init &&
        srcOp == ArraySource::Copy) {
      return language_cvw_arrayInitWithCopy(dest, src, count, stride, this);
    }

    if (this->hasLayoutString() && destOp == ArrayDest::Assign &&
        srcOp == ArraySource::Copy) {
      return language_cvw_arrayAssignWithCopy(dest, src, count, stride, this);
    }

    auto copy = get_witness_function<destOp, srcOp>(wtable);
    for (size_t i = 0; i < count; ++i) {
      auto offset = i * stride;
      auto *from = reinterpret_cast<OpaqueValue *>((char *)src + offset);
      auto *to = reinterpret_cast<OpaqueValue *>((char *)dest + offset);
      copy(to, from, this);
    }
    return;
  }

  // Back-to-front copy.
  assert(copyKind == ArrayCopy::BackToFront);
  assert(count != 0);

  auto copy = get_witness_function<destOp, srcOp>(wtable);
  size_t i = count;
  do {
    auto offset = --i * stride;
    auto *from = reinterpret_cast<OpaqueValue *>((char *)src + offset);
    auto *to = reinterpret_cast<OpaqueValue *>((char *)dest + offset);
    copy(to, from, this);
  } while (i != 0);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayInitWithCopy(OpaqueValue *dest, OpaqueValue *src, size_t count,
                             const Metadata *this) {
  array_copy_operation<ArrayDest::Init, ArraySource::Copy, ArrayCopy::NoAlias>(
      dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayInitWithTakeNoAlias(OpaqueValue *dest, OpaqueValue *src,
                                    size_t count, const Metadata *this) {
  array_copy_operation<ArrayDest::Init, ArraySource::Take, ArrayCopy::NoAlias>(
      dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayInitWithTakeFrontToBack(OpaqueValue *dest, OpaqueValue *src,
                                        size_t count, const Metadata *this) {
  array_copy_operation<ArrayDest::Init, ArraySource::Take,
                       ArrayCopy::FrontToBack>(dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayInitWithTakeBackToFront(OpaqueValue *dest, OpaqueValue *src,
                                        size_t count, const Metadata *this) {
  array_copy_operation<ArrayDest::Init, ArraySource::Take,
                       ArrayCopy::BackToFront>(dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayAssignWithCopyNoAlias(OpaqueValue *dest, OpaqueValue *src,
                                      size_t count, const Metadata *this) {
  array_copy_operation<ArrayDest::Assign, ArraySource::Copy,
                       ArrayCopy::NoAlias>(dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayAssignWithCopyFrontToBack(OpaqueValue *dest, OpaqueValue *src,
                                          size_t count, const Metadata *this) {
  array_copy_operation<ArrayDest::Assign, ArraySource::Copy,
                       ArrayCopy::FrontToBack>(dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayAssignWithCopyBackToFront(OpaqueValue *dest, OpaqueValue *src,
                                          size_t count, const Metadata *this) {
  array_copy_operation<ArrayDest::Assign, ArraySource::Copy,
                       ArrayCopy::BackToFront>(dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayAssignWithTake(OpaqueValue *dest, OpaqueValue *src,
                               size_t count, const Metadata *this) {
  array_copy_operation<ArrayDest::Assign, ArraySource::Take,
                       ArrayCopy::NoAlias>(dest, src, count, this);
}

LANGUAGE_RUNTIME_EXPORT
void language_arrayDestroy(OpaqueValue *begin, size_t count, const Metadata *this) {
  if (count == 0)
    return;

  auto wtable = this->getValueWitnesses();

  // Nothing to do if the type is POD.
  if (wtable->isPOD())
    return;

  auto stride = wtable->getStride();
  if (this->hasLayoutString()) {
    return language_cvw_arrayDestroy(begin, count, stride, this);
  }

  for (size_t i = 0; i < count; ++i) {
    auto offset = i * stride;
    auto *obj = reinterpret_cast<OpaqueValue *>((char *)begin + offset);
    wtable->destroy(obj, this);
  }
}
