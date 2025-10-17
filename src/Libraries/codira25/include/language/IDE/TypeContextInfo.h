/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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

//===--- TypeContextInfo.h --------------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_IDE_TYPECONTEXTINFO_H
#define LANGUAGE_IDE_TYPECONTEXTINFO_H

#include "language/AST/Type.h"
#include "language/Basic/Toolchain.h"

namespace language {
class IDEInspectionCallbacksFactory;

namespace ide {

/// A result item for context info query.
class TypeContextInfoItem {
public:
  /// Possible expected type.
  Type ExpectedTy;
  /// Members of \c ExpectedTy which can be referenced by "Implicit Member
  /// Expression".
  SmallVector<ValueDecl *, 0> ImplicitMembers;

  TypeContextInfoItem(Type ExpectedTy) : ExpectedTy(ExpectedTy) {}
};

/// An abstract base class for consumers of context info results.
class TypeContextInfoConsumer {
public:
  virtual ~TypeContextInfoConsumer() {}
  virtual void handleResults(ArrayRef<TypeContextInfoItem>) = 0;
};

/// Create a factory for code completion callbacks.
IDEInspectionCallbacksFactory *
makeTypeContextInfoCallbacksFactory(TypeContextInfoConsumer &Consumer);

} // namespace ide
} // namespace language

#endif // LANGUAGE_IDE_TYPECONTEXTINFO_H
