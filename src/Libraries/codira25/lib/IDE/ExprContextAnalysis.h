/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

//===--- ExprContextAnalysis.h - Expession context analysis ---------------===//
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

#ifndef LANGUAGE_IDE_EXPRCONTEXTANALYSIS_H
#define LANGUAGE_IDE_EXPRCONTEXTANALYSIS_H

#include "language/AST/Type.h"
#include "language/AST/TypeCheckRequests.h"
#include "language/AST/Types.h"
#include "language/Basic/Toolchain.h"
#include "language/Basic/SourceLoc.h"
#include "language/IDE/PossibleParamInfo.h"

namespace language {
class DeclContext;
class Expr;
class ValueDecl;

namespace ide {
enum class SemanticContextKind : uint8_t;

/// Type check parent contexts of the given decl context, and the body of the
/// given context until \c Loc if the context is a function body.
void typeCheckContextAt(TypeCheckASTNodeAtLocContext TypeCheckCtx,
                        SourceLoc Loc);

/// From \p DC, find and returns the outer most expression which source range is
/// exact the same as \p TargetRange. Returns \c nullptr if not found.
Expr *findParsedExpr(const DeclContext *DC, SourceRange TargetRange);

/// Collects possible expected return types of the given decl context.
/// \p DC should be an \c AbstractFunctionDecl or an \c AbstractClosureExpr.
void collectPossibleReturnTypesFromContext(DeclContext *DC,
                                           SmallVectorImpl<Type> &candidates);

struct FunctionTypeAndDecl {
  AnyFunctionType *Type;
  ValueDecl *Decl;
  std::optional<SemanticContextKind> SemanticContext;

  FunctionTypeAndDecl(AnyFunctionType *Type, ValueDecl *Decl)
      : Type(Type), Decl(Decl) {}
  FunctionTypeAndDecl(AnyFunctionType *Type, ValueDecl *Decl,
                      SemanticContextKind SemanticContext)
      : Type(Type), Decl(Decl), SemanticContext(SemanticContext) {}
};

} // namespace ide
} // namespace language

#endif // LANGUAGE_IDE_EXPRCONTEXTANALYSIS_H
