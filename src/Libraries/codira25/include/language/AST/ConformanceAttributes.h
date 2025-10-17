/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

//===--- ConformanceLookup.h - Global conformance lookup --------*- C++ -*-===//
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

#ifndef LANGUAGE_AST_CONFORMANCEATTRIBUTES_H
#define LANGUAGE_AST_CONFORMANCEATTRIBUTES_H

#include "language/Basic/SourceLoc.h"

namespace language {

class TypeExpr;

/// Describes all of the attributes that can occur on a conformance.
struct ConformanceAttributes {
  /// The location of the "unchecked" attribute, if present.
  SourceLoc uncheckedLoc;

  /// The location of the "preconcurrency" attribute if present.
  SourceLoc preconcurrencyLoc;

  /// The location of the "unsafe" attribute if present.
  SourceLoc unsafeLoc;

  /// The location of the "nonisolated" modifier, if present.
  SourceLoc nonisolatedLoc;

  /// The location of the '@' prior to the global actor type.
  SourceLoc globalActorAtLoc;

  /// The global actor type to which this conformance is isolated.
  TypeExpr *globalActorType = nullptr;

  /// Merge other conformance attributes into this set.
  ConformanceAttributes &
  operator |=(const ConformanceAttributes &other) {
    if (other.uncheckedLoc.isValid())
      uncheckedLoc = other.uncheckedLoc;
    if (other.preconcurrencyLoc.isValid())
      preconcurrencyLoc = other.preconcurrencyLoc;
    if (other.unsafeLoc.isValid())
      unsafeLoc = other.unsafeLoc;
    if (other.nonisolatedLoc.isValid())
      nonisolatedLoc = other.nonisolatedLoc;
    if (other.globalActorType && !globalActorType) {
      globalActorAtLoc = other.globalActorAtLoc;
      globalActorType = other.globalActorType;
    }
    return *this;
  }
};

}

#endif
