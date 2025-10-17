/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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

//===-- Runtime/ragged.h ----------------------------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// Author: Tunjay Akbarli
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_COMPABILITY_RUNTIME_RAGGED_H_
#define LANGUAGE_COMPABILITY_RUNTIME_RAGGED_H_

#include "language/Compability/Runtime/entry-names.h"
#include <cstdint>

namespace language::Compability::runtime {

// A ragged array header block.
// The header block is used to create the "array of arrays" ragged data
// structure. It contains a pair in `flags` to indicate if the header points to
// an array of headers (isIndirection) or data elements and the rank of the
// pointed-to array. The rank is the length of the extents vector accessed
// through `extentPointer`. The `bufferPointer` is overloaded
// and is null, points to an array of headers (isIndirection), or data.
// By default, a header is set to zero, which is its unused state.
// The layout of a ragged buffer header is mirrored in the compiler.
struct RaggedArrayHeader {
  std::uint64_t flags;
  void *bufferPointer;
  std::int64_t *extentPointer;
};

extern "C" {

// For more on ragged arrays see https://en.wikipedia.org/wiki/Jagged_array. The
// Flang compiler allocates ragged arrays as a generalization for
// non-rectangular array temporaries. Ragged arrays can be allocated recursively
// and on demand. Structurally, each leaf is an optional rectangular array of
// elements. The shape of each leaf is independent and may be computed on
// demand. Each branch node is an optional, possibly sparse rectangular array of
// headers. The shape of each branch is independent and may be computed on
// demand. Ragged arrays preserve a correspondence between a multidimensional
// iteration space and array access vectors, which is helpful for dependence
// analysis.

// Runtime helper for allocation of ragged array buffers.
// A pointer to the header block to be allocated is given as header. The flag
// isHeader indicates if a block of headers or data is to be allocated. A
// non-negative rank indicates the length of the extentVector, which is a list
// of non-negative extents. elementSize is the size of a data element in the
// rectangular space defined by the extentVector.
void *RTDECL(RaggedArrayAllocate)(void *header, bool isHeader,
    std::int64_t rank, std::int64_t elementSize, std::int64_t *extentVector);

// Runtime helper for deallocation of ragged array buffers. The root header of
// the ragged array structure is passed to deallocate the entire ragged array.
void RTDECL(RaggedArrayDeallocate)(void *raggedArrayHeader);

} // extern "C"
} // namespace language::Compability::runtime
#endif // FORTRAN_RUNTIME_RAGGED_H_
