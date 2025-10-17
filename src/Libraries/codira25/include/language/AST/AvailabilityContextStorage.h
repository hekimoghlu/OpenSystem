/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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

//===--- AvailabilityContextStorage.h - Codira AvailabilityContext ---------===//
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
// This file defines types used in the implementation of AvailabilityContext.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_AVAILABILITY_CONTEXT_STORAGE_H
#define LANGUAGE_AST_AVAILABILITY_CONTEXT_STORAGE_H

#include "language/AST/AvailabilityContext.h"
#include "toolchain/ADT/FoldingSet.h"
#include "toolchain/Support/TrailingObjects.h"

namespace language {

class DeclAvailabilityConstraints;

/// A wrapper for storing an `AvailabilityDomain` and its associated information
/// in `AvailabilityContext::Storage`.
class AvailabilityContext::DomainInfo final {
  AvailabilityDomain domain;
  AvailabilityRange range;

public:
  DomainInfo(AvailabilityDomain domain, const AvailabilityRange &range)
      : domain(domain), range(range) {};

  static DomainInfo unavailable(AvailabilityDomain domain) {
    return DomainInfo(domain, AvailabilityRange::neverAvailable());
  }

  AvailabilityDomain getDomain() const { return domain; }
  AvailabilityRange getRange() const { return range; }
  bool isUnavailable() const { return range.isKnownUnreachable(); }

  bool constrainRange(const AvailabilityRange &range);

  void Profile(toolchain::FoldingSetNodeID &ID) const {
    ID.AddPointer(domain.getOpaqueValue());
    range.getRawVersionRange().Profile(ID);
  }
};

/// As an implementation detail, the values that make up an `Availability`
/// context are uniqued and stored as folding set nodes.
class AvailabilityContext::Storage final
    : public toolchain::FoldingSetNode,
      public toolchain::TrailingObjects<Storage, DomainInfo> {
  friend TrailingObjects;

  Storage(const AvailabilityRange &platformRange, bool isDeprecated,
          unsigned domainInfoCount)
      : platformRange(platformRange), isDeprecated(isDeprecated),
        domainInfoCount(domainInfoCount) {};

public:
  /// The introduction version for the current platform.
  const AvailabilityRange platformRange;

  /// Whether or not the context is considered deprecated on the current
  /// platform.
  const unsigned isDeprecated : 1;

  /// The number of `DomainInfo` objects in trailing storage.
  const unsigned domainInfoCount;

  /// Retrieves the unique storage instance from the `ASTContext` for the given
  /// parameters.
  static const Storage *get(const AvailabilityRange &platformRange,
                            bool isDeprecated,
                            toolchain::ArrayRef<DomainInfo> domainInfos,
                            const ASTContext &ctx);

  toolchain::ArrayRef<DomainInfo> getDomainInfos() const {
    return toolchain::ArrayRef(getTrailingObjects<DomainInfo>(), domainInfoCount);
  }

  toolchain::SmallVector<DomainInfo, 4> copyDomainInfos() const {
    return toolchain::SmallVector<DomainInfo, 4>{getDomainInfos()};
  }

  /// Uniquing for `ASTContext`.
  static void Profile(toolchain::FoldingSetNodeID &ID,
                      const AvailabilityRange &platformRange, bool isDeprecated,
                      toolchain::ArrayRef<DomainInfo> domainInfos);

  void Profile(toolchain::FoldingSetNodeID &ID) const {
    Profile(ID, platformRange, static_cast<bool>(isDeprecated),
            getDomainInfos());
  }
};

} // end namespace language

#endif
