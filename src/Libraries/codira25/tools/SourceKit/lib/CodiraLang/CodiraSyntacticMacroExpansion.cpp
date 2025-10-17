/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

//===--- CodiraSyntaxMacro.cpp ---------------------------------------------===//
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

#include "CodiraLangSupport.h"
#include "language/AST/MacroDefinition.h"
#include "language/Frontend/Frontend.h"
#include "language/Frontend/PrintingDiagnosticConsumer.h"
#include "language/IDE/TypeContextInfo.h"
#include "language/IDETool/SyntacticMacroExpansion.h"
#include "language/Core/AST/ASTContext.h"
#include "language/Core/AST/Comment.h"
#include "language/Core/AST/Decl.h"

using namespace SourceKit;
using namespace language;
using namespace ide;

void CodiraLangSupport::expandMacroSyntactically(
    toolchain::MemoryBuffer *inputBuf, ArrayRef<const char *> args,
    ArrayRef<MacroExpansionInfo> reqExpansions,
    CategorizedEditsReceiver receiver) {

  std::string error;
  auto instance = SyntacticMacroExpansions->getInstance(args, inputBuf, error);
  if (!instance) {
    return receiver(
        RequestResult<ArrayRef<CategorizedEdits>>::fromError(error));
  }
  auto &ctx = instance->getASTContext();

  // Convert 'SourceKit::MacroExpansionInfo' to 'ide::MacroExpansionSpecifier'.
  SmallVector<ide::MacroExpansionSpecifier, 4> expansions;
  for (auto &req : reqExpansions) {
    unsigned offset = req.offset;

    language::MacroRoles macroRoles;
#define MACRO_ROLE(Name, Description)                   \
    if (req.roles.contains(SourceKit::MacroRole::Name)) \
      macroRoles |= language::MacroRole::Name;
#include "language/Basic/MacroRoles.def"

    MacroDefinition definition = [&] {
      if (auto *expanded =
              std::get_if<MacroExpansionInfo::ExpandedMacroDefinition>(
                  &req.macroDefinition)) {
        SmallVector<ExpandedMacroReplacement, 2> replacements;
        for (auto &reqReplacement : expanded->replacements) {
          replacements.push_back(
              {/*startOffset=*/reqReplacement.range.Offset,
               /*endOffset=*/reqReplacement.range.Offset +
                   reqReplacement.range.Length,
               /*parameterIndex=*/reqReplacement.parameterIndex});
        }
        SmallVector<ExpandedMacroReplacement, 2> genericReplacements;
        for (auto &genReqReplacement : expanded->genericReplacements) {
          genericReplacements.push_back(
              {/*startOffset=*/genReqReplacement.range.Offset,
               /*endOffset=*/genReqReplacement.range.Offset +
                   genReqReplacement.range.Length,
               /*parameterIndex=*/genReqReplacement.parameterIndex});
        }
        return MacroDefinition::forExpanded(ctx, expanded->expansionText,
                                            replacements, genericReplacements);
      } else if (auto *externalRef =
                     std::get_if<MacroExpansionInfo::ExternalMacroReference>(
                         &req.macroDefinition)) {
        return MacroDefinition::forExternal(
            ctx.getIdentifier(externalRef->moduleName),
            ctx.getIdentifier(externalRef->typeName));
      } else {
        return MacroDefinition::forUndefined();
      }
    }();

    expansions.push_back({offset, macroRoles, definition});
  }

  RequestRefactoringEditConsumer consumer(receiver);
  instance->expandAll(expansions, consumer);
  // consumer automatically send the results on destruction.
}
