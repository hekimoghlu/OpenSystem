/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

//===--- TypeResolutionStage.h - Type Resolution Stage ----------*- C++ -*-===//
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
#ifndef LANGUAGE_AST_TYPE_RESOLUTION_STAGE_H
#define LANGUAGE_AST_TYPE_RESOLUTION_STAGE_H

namespace toolchain {
class raw_ostream;
}

namespace language {

/// Describes the stage at which a particular type should be computed.
///
/// Later stages compute more information about the type, requiring more
/// complete analysis.
enum class TypeResolutionStage : uint8_t {
  /// Produces an interface type describing its structure, but without
  /// performing semantic analysis to resolve (e.g.) references to members of
  /// type parameters.
  Structural,

  /// Produces a complete interface type where all member references have been
  /// resolved.
  Interface,
};

/// Display a type resolution stage.
void simple_display(toolchain::raw_ostream &out, const TypeResolutionStage &value);

} // end namespace language

#endif // LANGUAGE_AST_TYPE_RESOLUTION_STAGE_H
