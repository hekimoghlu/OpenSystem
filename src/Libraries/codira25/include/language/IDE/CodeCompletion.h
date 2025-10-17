/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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

//===--- CodeCompletion.h - Routines for code completion --------*- C++ -*-===//
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

#ifndef LANGUAGE_IDE_CODECOMPLETION_H
#define LANGUAGE_IDE_CODECOMPLETION_H

#include "language/IDE/CodeCompletionConsumer.h"
#include "language/IDE/CodeCompletionContext.h"
#include "language/IDE/CodeCompletionResult.h"
#include "language/IDE/CodeCompletionResultSink.h"
#include "language/IDE/CompletionLookup.h"

namespace language {
class IDEInspectionCallbacksFactory;
class Decl;
class DeclContext;
class FrontendOptions;
class ModuleDecl;
class SourceFile;

namespace ide {

class CodeCompletionCache;
struct RequestedCachedModule;

/// A routine to remove code completion tokens from code completion
/// tests.
///
/// \code
/// code-completion-token:
///     '#^' identifier '^#'
/// \endcode
///
/// \param Input test source code.
/// \param TokenName names the token whose position should be returned in
/// \p CompletionOffset.
/// \param CompletionOffset set to ~0U on error, or to a 0-based byte offset on
/// success.
///
/// \returns test source code without any code completion tokens.
std::string removeCodeCompletionTokens(StringRef Input,
                                       StringRef TokenName,
                                       unsigned *CompletionOffset);

template <typename T>
ArrayRef<T> copyArray(toolchain::BumpPtrAllocator &Allocator,
                            ArrayRef<T> Arr) {
  T *Buffer = Allocator.Allocate<T>(Arr.size());
  std::copy(Arr.begin(), Arr.end(), Buffer);
  return toolchain::ArrayRef(Buffer, Arr.size());
}

bool isDynamicLookup(Type T);

void postProcessCompletionResults(
    MutableArrayRef<CodeCompletionResult *> results, CompletionKind Kind,
    const DeclContext *DC, CodeCompletionResultSink *Sink);

void collectCompletionResults(CodeCompletionContext &CompletionContext,
                              CompletionLookup &Lookup, DeclContext *DC,
                              const ExpectedTypeContext &TypeContext,
                              bool CanCurrDeclContextHandleAsync);

/// Create a factory for code completion callbacks.
IDEInspectionCallbacksFactory *
makeCodeCompletionCallbacksFactory(CodeCompletionContext &CompletionContext,
                                   CodeCompletionConsumer &Consumer);

/// Lookup the top-level code completions from \p module and store them in
/// \p targetSink.
///
/// Results are looked up as if in \p currDeclContext, which may be null.
void lookupCodeCompletionResultsFromModule(CodeCompletionResultSink &targetSink,
                                           const ModuleDecl *module,
                                           ArrayRef<std::string> accessPath,
                                           bool needLeadingDot,
                                           const SourceFile *SF);

void addExprKeywords(CodeCompletionResultSink &Sink, DeclContext *DC);

void addSuperKeyword(CodeCompletionResultSink &Sink, DeclContext *DC);

} // end namespace ide
} // end namespace language

#endif // LANGUAGE_IDE_CODECOMPLETION_H
