/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

//===--- SelectedOverloadInfo.h -------------------------------------------===//
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

#ifndef LANGUAGE_IDE_LANGUAGECOMPLETIONINFO_H
#define LANGUAGE_IDE_LANGUAGECOMPLETIONINFO_H

#include "language/AST/Decl.h"
#include "language/Sema/ConstraintSystem.h"

namespace language {
namespace ide {

using namespace language::constraints;

/// Information that \c getSelectedOverloadInfo gathered about a
/// \c SelectedOverload.
struct SelectedOverloadInfo {
  /// The function that is being called or the value that is being accessed.
  ConcreteDeclRef ValueRef;
  /// For a function, type of the called function itself (not its result type),
  /// for an arbitrary value the type of that value.
  Type ValueTy;
  /// The type on which the overload is being accessed. \c null if it does not
  /// have a base type, e.g. for a free function.
  Type BaseTy;

  ValueDecl *getValue() const { return ValueRef.getDecl(); }
};

/// Extract additional information about the overload that is being called by
/// \p CalleeLocator.
SelectedOverloadInfo getSelectedOverloadInfo(const Solution &S,
                                             ConstraintLocator *CalleeLocator);

} // end namespace ide
} // end namespace language

#endif // LANGUAGE_IDE_LANGUAGECOMPLETIONINFO_H
