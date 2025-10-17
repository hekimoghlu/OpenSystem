/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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

//===--- CompletionContextFinder.h ----------------------------------------===//
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

#ifndef LANGUAGE_SEMA_COMPLETIONCONTEXTFINDER_H
#define LANGUAGE_SEMA_COMPLETIONCONTEXTFINDER_H

#include "language/AST/ASTNode.h"
#include "language/AST/ASTWalker.h"
#include "language/AST/Expr.h"
#include "language/IDE/TypeCheckCompletionCallback.h"

namespace language {

namespace constraints {
class SyntacticElementTarget;
}

class CompletionContextFinder : public ASTWalker {
  enum class ContextKind {
    FallbackExpression,
    StringInterpolation,
    SingleStmtClosure,
    MultiStmtClosure,
    ErrorExpression
  };

  struct Context {
    ContextKind Kind;
    Expr *E;
  };

  /// Stack of all "interesting" contexts up to code completion expression.
  toolchain::SmallVector<Context, 4> Contexts;

  /// If we are completing inside an expression, the \c CodeCompletionExpr that
  /// represents the code completion token.

  /// The AST node that represents the code completion token, either as a
  /// \c CodeCompletionExpr or a \c KeyPathExpr which contains a code completion
  /// component.
  toolchain::PointerUnion<CodeCompletionExpr *, const KeyPathExpr *> CompletionNode;

  Expr *InitialExpr = nullptr;
  DeclContext *InitialDC;

  /// Whether we're looking for any viable fallback expression.
  bool ForFallback = false;

  /// Finder for fallback completion contexts within the outermost non-closure
  /// context of the code completion expression's direct context.
  CompletionContextFinder(DeclContext *completionDC)
    : InitialDC(completionDC), ForFallback(true) {
    while (auto *ACE = dyn_cast<AbstractClosureExpr>(InitialDC))
      InitialDC = ACE->getParent();
    InitialDC->walkContext(*this);
  }

public:
  MacroWalking getMacroWalkingBehavior() const override {
    return MacroWalking::Arguments;
  }

  /// Finder for completion contexts within the provided SyntacticElementTarget.
  CompletionContextFinder(constraints::SyntacticElementTarget target,
                          DeclContext *DC);

  static CompletionContextFinder forFallback(DeclContext *DC) {
    return CompletionContextFinder(DC);
  }

  PreWalkResult<Expr *> walkToExprPre(Expr *E) override;

  PostWalkResult<Expr *> walkToExprPost(Expr *E) override;

  PreWalkAction walkToDeclPre(Decl *D) override;

  bool locatedInStringInterpolation() const {
    return hasContext(ContextKind::StringInterpolation);
  }

  bool hasCompletionExpr() const {
    return CompletionNode.dyn_cast<CodeCompletionExpr *>() != nullptr;
  }

  CodeCompletionExpr *getCompletionExpr() const {
    assert(hasCompletionExpr());
    return CompletionNode.get<CodeCompletionExpr *>();
  }

  bool hasCompletionKeyPathComponent() const {
    return CompletionNode.dyn_cast<const KeyPathExpr *>() != nullptr;
  }

  bool hasCompletion() const {
    return !CompletionNode.isNull();
  }

  /// If we are completing in a key path, returns the \c KeyPath that contains
  /// the code completion component.
  const KeyPathExpr *getKeyPathContainingCompletionComponent() const {
    assert(hasCompletionKeyPathComponent());
    return CompletionNode.get<const KeyPathExpr *>();
  }

  /// If we are completing in a key path, returns the index at which the key
  /// path has the code completion component.
  size_t getKeyPathCompletionComponentIndex() const;

  struct Fallback {
    Expr *E;               ///< The fallback expression.
    DeclContext *DC;       ///< The fallback expression's decl context.
    bool SeparatePrecheck; ///< True if the fallback may require prechecking.
  };

  /// As a fallback sometimes its useful to not only type-check
  /// code completion expression directly but instead add some
  /// of the enclosing context e.g. when completion is an argument
  /// to a call.
  std::optional<Fallback> getFallbackCompletionExpr() const;

private:
  bool hasContext(ContextKind kind) const {
    return toolchain::find_if(Contexts, [&kind](const Context &currContext) {
             return currContext.Kind == kind;
           }) != Contexts.end();
  }
};


/// Returns \c true if \p range is valid and contains the IDE inspection
/// target. This performs the underlying check based on \c CharSourceRange
/// to make sure we correctly return \c true if the ide inspection target
/// is inside a string literal that's the last token in \p range.
bool containsIDEInspectionTarget(SourceRange range,
                                 const SourceManager &SourceMgr);

} // end namespace language

#endif // LANGUAGE_SEMA_COMPLETIONCONTEXTFINDER_H
