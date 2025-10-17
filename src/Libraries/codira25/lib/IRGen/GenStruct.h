/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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

//===--- GenStruct.h - Codira IR generation for structs ----------*- C++ -*-===//
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
//  This file provides the private interface to the struct-emission code.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENSTRUCT_H
#define LANGUAGE_IRGEN_GENSTRUCT_H

#include <optional>

namespace toolchain {
  class Constant;
}

namespace language {
  class CanType;
  class SILType;
  class VarDecl;

namespace irgen {
  class Address;
  class Explosion;
  class IRGenFunction;
  class IRGenModule;
  class MemberAccessStrategy;
  class TypeInfo;

  Address projectPhysicalStructMemberAddress(IRGenFunction &IGF,
                                             Address base,
                                             SILType baseType,
                                             VarDecl *field);
  void projectPhysicalStructMemberFromExplosion(IRGenFunction &IGF,
                                                SILType baseType,
                                                Explosion &base,
                                                VarDecl *field,
                                                Explosion &out);

  /// Return the constant offset of the given stored property in a struct,
  /// or return None if the field does not have fixed layout.
  toolchain::Constant *emitPhysicalStructMemberFixedOffset(IRGenModule &IGM,
                                                      SILType baseType,
                                                      VarDecl *field);

  /// Return a strategy for accessing the given stored struct property.
  ///
  /// This API is used by RemoteAST.
  MemberAccessStrategy
  getPhysicalStructMemberAccessStrategy(IRGenModule &IGM,
                                        SILType baseType, VarDecl *field);

  const TypeInfo *getPhysicalStructFieldTypeInfo(IRGenModule &IGM,
                                                 SILType baseType,
                                                 VarDecl *field);

  /// Returns the index of the element in the toolchain struct type which represents
  /// \p field in \p baseType.
  ///
  /// Returns None if \p field has an empty type and therefore has no
  /// corresponding element in the toolchain type.
  std::optional<unsigned> getPhysicalStructFieldIndex(IRGenModule &IGM,
                                                      SILType baseType,
                                                      VarDecl *field);
} // end namespace irgen
} // end namespace language

#endif
