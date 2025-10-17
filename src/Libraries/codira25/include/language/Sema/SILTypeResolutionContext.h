/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

//===--- SILTypeResolutionContext.h -----------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_SEMA_SILTYPERESOLUTIONCONTEXT_H
#define LANGUAGE_SEMA_SILTYPERESOLUTIONCONTEXT_H

#include "toolchain/ADT/DenseMap.h"
#include "language/Basic/UUID.h"

namespace language {
class GenericParamList;
class GenericEnvironment;

class SILTypeResolutionContext {
public:
  struct OpenedPackElement {
    SourceLoc DefinitionPoint;
    GenericParamList *Params;
    GenericEnvironment *Environment;
  };
  using OpenedPackElementsMap = toolchain::DenseMap<UUID, OpenedPackElement>;

  /// Are we requesting a SIL type?
  bool IsSILType;

  /// Look up types in the given parameter list.
  GenericParamList *GenericParams;

  /// Look up @pack_element environments in this map.
  OpenedPackElementsMap *OpenedPackElements;

  SILTypeResolutionContext(bool isSILType,
                           GenericParamList *genericParams,
                           OpenedPackElementsMap *openedPackElements)
    : IsSILType(isSILType),
      GenericParams(genericParams),
      OpenedPackElements(openedPackElements) {}
};

}

#endif
