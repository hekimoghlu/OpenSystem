/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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

//===--- AbstractConformance.h - Abstract conformance storage ---*- C++ -*-===//
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
// This file defines the AbstractConformance class, which represents
// the conformance of a type parameter or archetype to a protocol.
// These are usually stashed inside a ProtocolConformanceRef.
//
//===----------------------------------------------------------------------===//
#ifndef LANGUAGE_AST_ABSTRACT_CONFORMANCE_H
#define LANGUAGE_AST_ABSTRACT_CONFORMANCE_H

#include "language/AST/Type.h"
#include "toolchain/ADT/FoldingSet.h"

namespace language {
class AbstractConformance final : public toolchain::FoldingSetNode {
  Type conformingType;
  ProtocolDecl *requirement;

public:
  AbstractConformance(Type conformingType, ProtocolDecl *requirement)
    : conformingType(conformingType), requirement(requirement) { }

  Type getType() const { return conformingType; }
  ProtocolDecl *getProtocol() const { return requirement; }

  void Profile(toolchain::FoldingSetNodeID &id) const {
    Profile(id, getType(), getProtocol());
  }

  /// Profile the storage for this conformance, for use with LLVM's FoldingSet.
  static void Profile(toolchain::FoldingSetNodeID &id,
                      Type conformingType,
                      ProtocolDecl *requirement) {
    id.AddPointer(conformingType.getPointer());
    id.AddPointer(requirement);
  }
};

}

#endif // LANGUAGE_AST_ABSTRACT_CONFORMANCE_H

