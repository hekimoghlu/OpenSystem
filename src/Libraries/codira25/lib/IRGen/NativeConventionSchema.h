/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

//===--- NativeConventionSchema.h - R-Value Schema for CodiraCC --*- C++ -*-===//
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
// A schema that describes the explosion of values for passing according to the
// native calling convention.
//
//===----------------------------------------------------------------------===//
#ifndef LANGUAGE_IRGEN_NATIVECONVENTIONSCHEMA_H
#define LANGUAGE_IRGEN_NATIVECONVENTIONSCHEMA_H

#include "language/Core/CodeGen/CodiraCallingConv.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include "IRGen.h"
#include "IRGenFunction.h"

namespace language {
namespace irgen {

using CodiraAggLowering = language::Core::CodeGen::languagecall::CodiraAggLowering;

class NativeConventionSchema {
  CodiraAggLowering Lowering;
  bool RequiresIndirect;

public:
  using EnumerationCallback = CodiraAggLowering::EnumerationCallback;

  NativeConventionSchema(IRGenModule &IGM, const TypeInfo *TI, bool isResult);

  NativeConventionSchema() = delete;
  NativeConventionSchema(const NativeConventionSchema &) = delete;
  NativeConventionSchema &operator=(const NativeConventionSchema&) = delete;

  bool requiresIndirect() const { return RequiresIndirect; }
  bool shouldReturnTypedErrorIndirectly() const {
    return requiresIndirect() || Lowering.shouldReturnTypedErrorIndirectly();
  }
  bool empty() const { return Lowering.empty(); }

  toolchain::Type *getExpandedType(IRGenModule &IGM) const;

  /// The number of components in the schema.
  unsigned size() const;

  void enumerateComponents(EnumerationCallback callback) const {
    Lowering.enumerateComponents(callback);
  }

  /// Map from a non-native explosion to an explosion that follows the native
  /// calling convention's schema.
  Explosion mapIntoNative(IRGenModule &IGM, IRGenFunction &IGF,
                          Explosion &fromNonNative, SILType type,
                          bool isOutlined, bool mayPeepholeLoad = false) const;

  /// Map form a native explosion that follows the native calling convention's
  /// schema to a non-native explosion whose schema is described by
  /// type.getSchema().
  Explosion mapFromNative(IRGenModule &IGM, IRGenFunction &IGF,
                          Explosion &native, SILType type) const;

  /// Return a pair of structs that can be used to load/store the components of
  /// the native schema from/to the memory representation as defined by the
  /// value's loadable type info.
  /// The second layout is only necessary if there are overlapping components in
  /// the legal type sequence. It contains the non-integer components of
  /// overlapped components of the legal type sequence.
  ///
  /// \p ExpandedTyIndices is a map from the non-array type elements of the
  /// returned struct types (viewed concatenated) to the index in the expanded
  /// type.
  std::pair<toolchain::StructType *, toolchain::StructType *>
  getCoercionTypes(IRGenModule &IGM,
                   SmallVectorImpl<unsigned> &expandedTyIndicesMap) const;
};


} // end namespace irgen
} // end namespace language

#endif
