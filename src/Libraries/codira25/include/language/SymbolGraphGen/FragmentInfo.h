/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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

//===--- FragmentInfo.h - Codira SymbolGraph Declaration Fragment Info -----===//
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

#ifndef LANGUAGE_SYMBOLGRAPHGEN_FRAGMENTINFO_H
#define LANGUAGE_SYMBOLGRAPHGEN_FRAGMENTINFO_H

#include "toolchain/ADT/SmallVector.h"
#include "PathComponent.h"

namespace language {
class ValueDecl;

namespace symbolgraphgen {

/// Summary information for a symbol referenced in a symbol graph declaration fragment.
struct FragmentInfo {
  /// The language decl of the referenced symbol.
  const ValueDecl *VD;
  /// The path components of the referenced symbol.
  SmallVector<PathComponent, 4> ParentContexts;
};

} // end namespace symbolgraphgen
} // end namespace language

#endif // LANGUAGE_SYMBOLGRAPHGEN_FRAGMENTINFO_H
