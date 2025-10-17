/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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

//===--- GenArchetype.h - Codira IR generation for archetypes ----*- C++ -*-===//
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
//  This file provides the private interface to the archetype emission code.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENARCHETYPE_H
#define LANGUAGE_IRGEN_GENARCHETYPE_H

#include "language/AST/Types.h"
#include "toolchain/ADT/STLExtras.h"

namespace toolchain {
  class Value;
}

namespace language {
  class AssociatedTypeDecl;
  class ProtocolDecl;
  class SILType;

namespace irgen {
  class Address;
  class IRGenFunction;
  class DynamicMetadataRequest;
  class MetadataResponse;

  using GetTypeParameterInContextFn =
    toolchain::function_ref<CanType(CanType type)>;

  /// Emit a type metadata reference for an archetype.
  MetadataResponse emitArchetypeTypeMetadataRef(IRGenFunction &IGF,
                                                CanArchetypeType archetype,
                                                DynamicMetadataRequest request);

  /// Emit a witness table reference.
  toolchain::Value *emitArchetypeWitnessTableRef(IRGenFunction &IGF,
                                            CanArchetypeType archetype,
                                            ProtocolDecl *protocol);

  /// Emit a metadata reference for an associated type of an archetype.
  MetadataResponse emitAssociatedTypeMetadataRef(IRGenFunction &IGF,
                                                 CanArchetypeType origin,
                                                 AssociatedTypeDecl *assocType,
                                                 DynamicMetadataRequest request);

  /// Emit a dynamic metatype lookup for the given archetype.
  toolchain::Value *emitDynamicTypeOfOpaqueArchetype(IRGenFunction &IGF,
                                                Address archetypeAddr,
                                                SILType archetypeType);
  
  /// Emit a lookup for an opaque result type's metadata.
  MetadataResponse emitOpaqueTypeMetadataRef(IRGenFunction &IGF,
                                             CanOpaqueTypeArchetypeType archetype,
                                             DynamicMetadataRequest request);
  /// Emit a lookup for an opaque result type's protocol conformance.
  toolchain::Value *emitOpaqueTypeWitnessTableRef(IRGenFunction &IGF,
                                             CanOpaqueTypeArchetypeType archetype,
                                             ProtocolDecl *protocol);
} // end namespace irgen
} // end namespace language

#endif
