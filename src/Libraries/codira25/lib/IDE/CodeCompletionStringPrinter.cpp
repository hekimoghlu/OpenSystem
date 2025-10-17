/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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

//===--- CodeCompletionStringPrinter.cpp ----------------------------------===//
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

#include "language/IDE/CodeCompletionStringPrinter.h"
#include "CodeCompletionResultBuilder.h"
#include "language/AST/Module.h"
#include "language/Basic/Assertions.h"

using namespace language;
using namespace language::ide;

using ChunkKind = CodeCompletionString::Chunk::ChunkKind;

std::optional<ChunkKind>
CodeCompletionStringPrinter::getChunkKindForPrintNameContext(
    PrintNameContext context) const {
  switch (context) {
  case PrintNameContext::Keyword:
    if (isCurrentStructureKind(PrintStructureKind::EffectsSpecifiers)) {
      return ChunkKind::EffectsSpecifierKeyword;
    }
    return ChunkKind::Keyword;
  case PrintNameContext::IntroducerKeyword:
    return ChunkKind::DeclIntroducer;
  case PrintNameContext::Attribute:
    return ChunkKind::Attribute;
  case PrintNameContext::FunctionParameterExternal:
    if (isInType()) {
      return std::nullopt;
    }
    return ChunkKind::ParameterDeclExternalName;
  case PrintNameContext::FunctionParameterLocal:
    if (isInType()) {
      return std::nullopt;
    }
    return ChunkKind::ParameterDeclLocalName;
  default:
    return std::nullopt;
  }
}

std::optional<ChunkKind>
CodeCompletionStringPrinter::getChunkKindForStructureKind(
    PrintStructureKind Kind) const {
  switch (Kind) {
  case PrintStructureKind::FunctionParameter:
    if (isInType()) {
      return std::nullopt;
    }
    return ChunkKind::ParameterDeclBegin;
  case PrintStructureKind::DefaultArgumentClause:
    return ChunkKind::DefaultArgumentClauseBegin;
  case PrintStructureKind::DeclGenericParameterClause:
    return ChunkKind::GenericParameterClauseBegin;
  case PrintStructureKind::DeclGenericRequirementClause:
    return ChunkKind::GenericRequirementClauseBegin;
  case PrintStructureKind::EffectsSpecifiers:
    return ChunkKind::EffectsSpecifierClauseBegin;
  case PrintStructureKind::DeclResultTypeClause:
    return ChunkKind::DeclResultTypeClauseBegin;
  case PrintStructureKind::FunctionParameterType:
    return ChunkKind::ParameterDeclTypeBegin;
  default:
    return std::nullopt;
  }
}

void CodeCompletionStringPrinter::startNestedGroup(ChunkKind Kind) {
  flush();
  Builder.CurrentNestingLevel++;
  Builder.addSimpleChunk(Kind);
}

void CodeCompletionStringPrinter::endNestedGroup() {
  flush();
  Builder.CurrentNestingLevel--;
}

void CodeCompletionStringPrinter::flush() {
  if (Buffer.empty())
    return;
  Builder.addChunkWithText(CurrChunkKind, Buffer);
  Buffer.clear();
}

/// Start \c AttributeAndModifierListBegin group. This must be called before
/// any attributes/modifiers printed to the output when printing an override
/// compleion.
void CodeCompletionStringPrinter::startPreamble() {
  assert(!InPreamble);
  startNestedGroup(ChunkKind::AttributeAndModifierListBegin);
  InPreamble = true;
}

void CodeCompletionStringPrinter::endPremable() {
  if (!InPreamble)
    return;
  InPreamble = false;
  endNestedGroup();
}

void CodeCompletionStringPrinter::printText(StringRef Text) {
  // Detect ': ' and ', ' in parameter clauses.
  // FIXME: Is there a better way?
  if (isCurrentStructureKind(PrintStructureKind::FunctionParameter) &&
      Text == ": ") {
    setNextChunkKind(ChunkKind::ParameterDeclColon);
  } else if (isCurrentStructureKind(
                 PrintStructureKind::FunctionParameterList) &&
             Text == ", ") {
    setNextChunkKind(ChunkKind::Comma);
  }

  if (CurrChunkKind != NextChunkKind) {
    // If the next desired kind is different from the current buffer, flush
    // the current buffer.
    flush();
    CurrChunkKind = NextChunkKind;
  }
  Buffer.append(Text);
}

void CodeCompletionStringPrinter::printTypeRef(Type T, const TypeDecl *TD,
                                               Identifier Name,
                                               PrintNameContext NameContext) {

  NextChunkKind = TD->getModuleContext()->isNonUserModule()
                      ? ChunkKind::TypeIdSystem
                      : ChunkKind::TypeIdUser;

  ASTPrinter::printTypeRef(T, TD, Name, NameContext);
  NextChunkKind = ChunkKind::Text;
}

void CodeCompletionStringPrinter::printDeclLoc(const Decl *D) {
  endPremable();
  setNextChunkKind(ChunkKind::BaseName);
}

void CodeCompletionStringPrinter::printDeclNameEndLoc(const Decl *D) {
  setNextChunkKind(ChunkKind::Text);
}

void CodeCompletionStringPrinter::printNamePre(PrintNameContext context) {
  if (context == PrintNameContext::IntroducerKeyword)
    endPremable();
  if (auto Kind = getChunkKindForPrintNameContext(context))
    setNextChunkKind(*Kind);
}

void CodeCompletionStringPrinter::printNamePost(PrintNameContext context) {
  if (getChunkKindForPrintNameContext(context))
    setNextChunkKind(ChunkKind::Text);
}

void CodeCompletionStringPrinter::printTypePre(const TypeLoc &TL) {
  ++TypeDepth;
}

void CodeCompletionStringPrinter::printTypePost(const TypeLoc &TL) {
  assert(TypeDepth > 0);
  --TypeDepth;
}

void CodeCompletionStringPrinter::printStructurePre(PrintStructureKind Kind,
                                                    const Decl *D) {
  StructureStack.push_back(Kind);

  if (auto chunkKind = getChunkKindForStructureKind(Kind))
    startNestedGroup(*chunkKind);
}

void CodeCompletionStringPrinter::printStructurePost(PrintStructureKind Kind,
                                                     const Decl *D) {
  if (getChunkKindForStructureKind(Kind))
    endNestedGroup();

  assert(Kind == StructureStack.back());
  StructureStack.pop_back();
}
