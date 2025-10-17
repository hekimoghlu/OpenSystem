/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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

//===--- TypeWalker.h - Class for walking a Type ----------------*- C++ -*-===//
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

#ifndef LANGUAGE_AST_TYPEWALKER_H
#define LANGUAGE_AST_TYPEWALKER_H

#include "language/AST/Type.h"

namespace language {

/// An abstract class used to traverse a Type.
class TypeWalker {
public:
  enum class Action {
    Continue,
    SkipNode,
    Stop
  };

  /// This method is called when first visiting a type before walking into its
  /// children.
  virtual Action walkToTypePre(Type ty) { return Action::Continue; }

  /// This method is called after visiting a type's children.
  virtual Action walkToTypePost(Type ty) { return Action::Continue; }

protected:
  TypeWalker() = default;
  TypeWalker(const TypeWalker &) = default;
  virtual ~TypeWalker() = default;
  
  virtual void anchor();
};

} // end namespace language

#endif
