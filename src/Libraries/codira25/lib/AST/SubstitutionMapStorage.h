/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

//===--- SubstitutionMapStorage.h - Substitution Map Storage ----*- C++ -*-===//
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
// This file defines the SubstitutionMap::Storage class, which is used as the
// backing storage for SubstitutionMap.
//
//===----------------------------------------------------------------------===//
#ifndef LANGUAGE_AST_SUBSTITUTION_MAP_STORAGE_H
#define LANGUAGE_AST_SUBSTITUTION_MAP_STORAGE_H

#include "language/AST/DiagnosticEngine.h"
#include "language/AST/DiagnosticsCommon.h"
#include "language/AST/ExistentialLayout.h"
#include "language/AST/FileSystem.h"
#include "language/AST/GenericSignature.h"
#include "language/AST/SubstitutionMap.h"
#include "toolchain/ADT/FoldingSet.h"
#include "toolchain/Support/FileSystem.h"
#include "toolchain/Support/TrailingObjects.h"

namespace language {

class SubstitutionMap::Storage final
  : public toolchain::FoldingSetNode,
    toolchain::TrailingObjects<Storage, Type, ProtocolConformanceRef>
{
  friend TrailingObjects;

  /// The generic signature for which we are performing substitutions.
  GenericSignature genericSig;

  /// The number of conformance requirements, cached to avoid constantly
  /// recomputing it on conformance-buffer access.
  const unsigned numConformanceRequirements;

  Storage() = delete;

  Storage(GenericSignature genericSig,
          ArrayRef<Type> replacementTypes,
          ArrayRef<ProtocolConformanceRef> conformances);

  friend class SubstitutionMap;

private:
  unsigned getNumReplacementTypes() const {
    return genericSig.getGenericParams().size();
  }

  size_t numTrailingObjects(OverloadToken<Type>) const {
    return getNumReplacementTypes();
  }

  size_t numTrailingObjects(OverloadToken<ProtocolConformanceRef>) const {
    return numConformanceRequirements;
  }

public:
  /// Form storage for the given generic signature and its replacement
  /// types and conformances.
  static Storage *get(GenericSignature genericSig,
                      ArrayRef<Type> replacementTypes,
                      ArrayRef<ProtocolConformanceRef> conformances);

  /// Retrieve the generic signature that describes the shape of this
  /// storage.
  GenericSignature getGenericSignature() const { return genericSig; }

  /// Retrieve the array of replacement types, which line up with the
  /// generic parameters.
  ///
  /// Note that the types may be null, for cases where the generic parameter
  /// is concrete but hasn't been queried yet.
  ArrayRef<Type> getReplacementTypes() const {
    return toolchain::ArrayRef(getTrailingObjects<Type>(), getNumReplacementTypes());
  }

  MutableArrayRef<Type> getReplacementTypes() {
    return MutableArrayRef<Type>(getTrailingObjects<Type>(),
                                 getNumReplacementTypes());
  }

  /// Retrieve the array of protocol conformances, which line up with the
  /// requirements of the generic signature.
  ArrayRef<ProtocolConformanceRef> getConformances() const {
    return toolchain::ArrayRef(getTrailingObjects<ProtocolConformanceRef>(),
                          numConformanceRequirements);
  }
  MutableArrayRef<ProtocolConformanceRef> getConformances() {
    return MutableArrayRef<ProtocolConformanceRef>(
                              getTrailingObjects<ProtocolConformanceRef>(),
                              numConformanceRequirements);
  }

  /// Profile the substitution map storage, for use with LLVM's FoldingSet.
  void Profile(toolchain::FoldingSetNodeID &id) const {
    Profile(id, getGenericSignature(), getReplacementTypes(),
            getConformances());
  }

  /// Profile the substitution map storage, for use with LLVM's FoldingSet.
  static void Profile(toolchain::FoldingSetNodeID &id,
                      GenericSignature genericSig,
                      ArrayRef<Type> replacementTypes,
                      ArrayRef<ProtocolConformanceRef> conformances);
};

}

#endif // LANGUAGE_AST_SUBSTITUTION_MAP_STORAGE_H
