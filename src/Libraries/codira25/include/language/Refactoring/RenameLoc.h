/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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

//===----------------------------------------------------------------------===//
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

#ifndef LANGUAGE_REFACTORING_RENAMELOC_H
#define LANGUAGE_REFACTORING_RENAMELOC_H

#include "language/Basic/Toolchain.h"

namespace language {
namespace ide {

/// Describes how a `ResolvedLoc` is being used
enum class RenameLocUsage {
  /// The definition of a function/subscript/variable/...
  Definition,

  /// The symbol is being referenced.
  ///
  /// This includes
  ///  - References to variables
  ///  - Unapplied references to functions (`myStruct.memberFunc`)
  ///  - Calls to subscripts (`myArray[1]`, location is `[` here, length 1)
  Reference,

  /// A function that is being called.
  Call,

  /// Unknown name usage occurs if we don't have an entry in the index that
  /// tells us whether the location is a call, reference or a definition. The
  /// most common reasons why this happens is if the editor is adding syntactic
  /// results (eg. from comments or string literals).
  Unknown
};

/// The input to `findSyntacticRenameRanges`.
///
/// Specifies the location of a base name for which `findSyntacticRenameRanges`
/// should find the argument labels as well as some semantic information needed
/// to resolve the rename ranges.
struct RenameLoc {
  /// The line at which the base name starts (1-based).
  unsigned Line;

  /// The column at which the base name (excluding trivia) starts (1-based).
  unsigned Column;

  /// The offset at which the related name starts.
  unsigned Offset;

  /// The length of the base name in the related identifier. For functions,
  /// this does not include the parameters/arguments.
  unsigned Length;

  /// How the identifier is being used (call, reference, definition, unknown).
  ///
  /// Used to decide whether a given occurance should be renamed and/or if its
  /// argument labels should be renamed.
  RenameLocUsage Usage;

  /// The old decl name being renamed.
  ///
  /// ### Examples
  /// - `foo(a:b:)` for a function with two parameters.
  /// - `foo` for a variable.
  /// - `foo(_:)` for a function with a single unnamed parameter
  /// - `foo()` for a function without parameters.
  StringRef OldName;

  RenameLoc(unsigned Line, unsigned Column, RenameLocUsage Usage,
            StringRef OldName)
      : Line(Line), Column(Column), Usage(Usage), OldName(OldName) {}
};

} // namespace ide
} // namespace language

#endif // LANGUAGE_REFACTORING_RENAMELOC_H
