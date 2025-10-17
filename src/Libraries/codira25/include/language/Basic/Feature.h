/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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

//===--- Feature.h - Helpers related to Codira features ----------*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_FEATURE_H
#define LANGUAGE_BASIC_FEATURE_H

#include "language/Basic/Toolchain.h"

#include "toolchain/ADT/StringRef.h"
#include <optional>

namespace language {

class LangOptions;

/// Enumeration describing all of the named features.
struct Feature {
  enum class InnerKind : uint16_t {
#define LANGUAGE_FEATURE(FeatureName, SENumber, Description) FeatureName,
#include "language/Basic/Features.def"
  };

  InnerKind kind;

  constexpr Feature(InnerKind kind) : kind(kind) {}
  constexpr Feature(unsigned inputKind) : kind(InnerKind(inputKind)) {}

  constexpr operator InnerKind() const { return kind; }
  constexpr explicit operator unsigned() const { return unsigned(kind); }
  constexpr explicit operator size_t() const { return size_t(kind); }

  static constexpr unsigned getNumFeatures() {
    enum Features {
#define LANGUAGE_FEATURE(FeatureName, SENumber, Description) FeatureName,
#include "language/Basic/Features.def"
      NumFeatures
    };
    return NumFeatures;
  }

#define LANGUAGE_FEATURE(FeatureName, SENumber, Description)                   \
  static const Feature FeatureName;
#include "language/Basic/Features.def"

  /// Check whether the given feature is available in production compilers.
  bool isAvailableInProduction() const;

  /// Determine the in-source name of the given feature.
  toolchain::StringRef getName() const;

  /// Determine whether the given feature supports migration mode.
  bool isMigratable() const;

  /// Determine whether this feature should be included in the
  /// module interface
  bool includeInModuleInterface() const;

  /// Determine whether the first feature is more recent (and thus implies
  /// the existence of) the second feature.  Only meaningful for suppressible
  /// features.
  constexpr bool featureImpliesFeature(Feature implied) const {
    // Suppressible features are expected to be listed in order of
    // addition in Features.def.
    return (unsigned)kind < (unsigned)implied.kind;
  }

  /// Get the feature corresponding to this "future" feature, if there is one.
  static std::optional<Feature> getUpcomingFeature(StringRef name);

  /// Get the feature corresponding to this "experimental" feature, if there is
  /// one.
  static std::optional<Feature> getExperimentalFeature(StringRef name);

  /// Get the major language version in which this feature was introduced, or
  /// \c None if it does not have such a version.
  std::optional<unsigned> getLanguageVersion() const;
};

#define LANGUAGE_FEATURE(FeatureName, SENumber, Description)                   \
  constexpr inline Feature Feature::FeatureName =                              \
      Feature::InnerKind::FeatureName;
#include "language/Basic/Features.def"
}

#endif // LANGUAGE_BASIC_FEATURES_H
