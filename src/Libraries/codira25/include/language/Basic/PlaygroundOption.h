/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

//===--- PlaygroundOption.h - Playground Transform Options -----*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_PLAYGROUND_OPTIONS_H
#define LANGUAGE_BASIC_PLAYGROUND_OPTIONS_H

#include "toolchain/ADT/SmallSet.h"
#include "toolchain/ADT/StringRef.h"
#include <optional>

namespace language {

/// Enumeration describing all of the available playground options.
enum class PlaygroundOption {
#define PLAYGROUND_OPTION(OptionName, Description, DefaultOn, HighPerfOn) \
  OptionName,
#include "language/Basic/PlaygroundOptions.def"
};

constexpr unsigned numPlaygroundOptions() {
  enum PlaygroundOptions {
#define PLAYGROUND_OPTION(OptionName, Description, DefaultOn, HighPerfOn) \
    OptionName,
#include "language/Basic/PlaygroundOptions.def"
    NumPlaygroundOptions
  };
  return NumPlaygroundOptions;
}

/// Return the name of the given playground option.
toolchain::StringRef getPlaygroundOptionName(PlaygroundOption option);

/// Get the playground option that corresponds to a given name, if there is one.
std::optional<PlaygroundOption> getPlaygroundOption(toolchain::StringRef name);

/// Set of enabled playground options.
typedef toolchain::SmallSet<PlaygroundOption, 8> PlaygroundOptionSet;

}

#endif // LANGUAGE_BASIC_PLAYGROUND_OPTIONS_H
