/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

//===--- GenCast.h - Codira IR generation for dynamic casts ------*- C++ -*-===//
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
//  This file provides the private interface to the dynamic cast code.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENCAST_H
#define LANGUAGE_IRGEN_GENCAST_H

#include "language/AST/Types.h"

namespace toolchain {
  class Value;
  class BasicBlock;
}

namespace language {
  class SILType;
  class ProtocolDecl;
  enum class CastConsumptionKind : unsigned char;
  class CheckedCastInstOptions;

namespace irgen {
  class Address;
  class IRGenFunction;
  class Explosion;

  /// Discriminator for checked cast modes.
  enum class CheckedCastMode : uint8_t {
    Unconditional,
    Conditional,
  };

  toolchain::Value *emitCheckedCast(IRGenFunction &IGF,
                               Address src,
                               CanType fromType,
                               Address dest,
                               CanType toType,
                               CastConsumptionKind consumptionKind,
                               CheckedCastMode mode,
                               CheckedCastInstOptions options);

  void emitScalarCheckedCast(IRGenFunction &IGF, Explosion &value,
                             SILType sourceLoweredType,
                             CanType sourceFormalType,
                             SILType targetLoweredType,
                             CanType targetFormalType,
                             CheckedCastMode mode,
                             CheckedCastInstOptions options,
                             Explosion &out);

  toolchain::Value *emitFastClassCastIfPossible(
      IRGenFunction &IGF, toolchain::Value *instance, CanType sourceFormalType,
      CanType targetFormalType, CheckedCastMode mode, bool sourceWrappedInOptional,
      toolchain::BasicBlock *&nilCheckBB, toolchain::BasicBlock *&nilMergeBB);

  /// Convert a class object to the given destination type,
  /// using a runtime-checked cast.
  toolchain::Value *emitClassDowncast(IRGenFunction &IGF,
                                 toolchain::Value *from,
                                 CanType toType,
                                 CheckedCastMode mode);

  /// A result of a cast generation function.
  struct FailableCastResult {
    /// An i1 value that's set to True if the cast succeeded.
    toolchain::Value *succeeded;
    /// On success, this value stores the result of the cast operation.
    toolchain::Value *casted;
  };

  /// Convert the given value to the exact destination type.
  FailableCastResult emitClassIdenticalCast(IRGenFunction &IGF,
                                            toolchain::Value *from,
                                            SILType fromType,
                                            SILType toType);

  /// Emit a checked cast of a metatype.
  void emitMetatypeDowncast(IRGenFunction &IGF,
                            toolchain::Value *metatype,
                            CanMetatypeType toMetatype,
                            CheckedCastMode mode,
                            Explosion &ex);

  /// Emit a checked cast to a class-constrained protocol or protocol
  /// composition.
  ///
  /// If a metatype kind is provided, the cast is done as a metatype cast. If
  /// not, the cast is done as a class instance cast.
  void emitScalarExistentialDowncast(
      IRGenFunction &IGF, toolchain::Value *orig, SILType srcType, SILType destType,
      CheckedCastMode mode, bool sourceWrappedInOptional,
      std::optional<MetatypeRepresentation> metatypeKind, Explosion &ex);

  /// Emit a checked cast from a metatype to AnyObject.
  toolchain::Value *emitMetatypeToAnyObjectDowncast(IRGenFunction &IGF,
                                            toolchain::Value *metatypeValue,
                                            CanAnyMetatypeType type,
                                            CheckedCastMode mode);

  /// Emit a Protocol* value referencing an ObjC protocol.
  toolchain::Value *emitReferenceToObjCProtocol(IRGenFunction &IGF,
                                           ProtocolDecl *proto);
} // end namespace irgen
} // end namespace language

#endif
