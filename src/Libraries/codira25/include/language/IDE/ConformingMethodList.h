/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

//===--- ConformingMethodList.h --- -----------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_IDE_CONFORMINGMETHODLIST_H
#define LANGUAGE_IDE_CONFORMINGMETHODLIST_H

#include "language/AST/Type.h"
#include "language/Basic/Toolchain.h"

namespace language {
class IDEInspectionCallbacksFactory;

namespace ide {

/// A result item for context info query.
class ConformingMethodListResult {
public:
  /// The decl context of the parsed expression.
  DeclContext *DC;

  /// The resolved type of the expression.
  Type ExprType;

  /// Methods which satisfy the criteria.
  SmallVector<ValueDecl *, 0> Members;

  ConformingMethodListResult(DeclContext *DC, Type ExprType)
      : DC(DC), ExprType(ExprType) {}
};

/// An abstract base class for consumers of context info results.
class ConformingMethodListConsumer {
public:
  virtual ~ConformingMethodListConsumer() {}
  virtual void handleResult(const ConformingMethodListResult &result) = 0;
};

/// Create a factory for code completion callbacks.
IDEInspectionCallbacksFactory *makeConformingMethodListCallbacksFactory(
    ArrayRef<const char *> expectedTypeNames,
    ConformingMethodListConsumer &Consumer);

} // namespace ide
} // namespace language

#endif // LANGUAGE_IDE_CONFORMINGMETHODLIST_H
