/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

//===--- TestContext.h - Helper for setting up ASTContexts ------*- C++ -*-===//
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

#include "language/AST/ASTContext.h"
#include "language/AST/DiagnosticEngine.h"
#include "language/AST/Module.h"
#include "language/AST/SourceFile.h"
#include "language/Basic/LangOptions.h"
#include "language/Basic/SourceManager.h"
#include "language/SymbolGraphGen/SymbolGraphOptions.h"

#include "toolchain/TargetParser/Host.h"

namespace language {
namespace unittest {

/// Helper class used to set the LangOpts target before initializing the
/// ASTContext.
///
/// \see TestContext
class TestContextBase {
public:
  LangOptions LangOpts;
  TypeCheckerOptions TypeCheckerOpts;
  SILOptions SILOpts;
  SearchPathOptions SearchPathOpts;
  ClangImporterOptions ClangImporterOpts;
  symbolgraphgen::SymbolGraphOptions SymbolGraphOpts;
  CASOptions CASOpts;
  SerializationOptions SerializationOpts;
  SourceManager SourceMgr;
  DiagnosticEngine Diags;

  TestContextBase(toolchain::Triple target) : Diags(SourceMgr) {
    LangOpts.Target = target;
  }
};

enum ShouldDeclareOptionalTypes : bool {
  DoNotDeclareOptionalTypes,
  DeclareOptionalTypes
};

/// Owns an ASTContext and the associated types.
class TestContext : public TestContextBase {
  SourceFile *FileForLookups;

public:
  ASTContext &Ctx;

  TestContext(
      ShouldDeclareOptionalTypes optionals = DoNotDeclareOptionalTypes,
      toolchain::Triple target = toolchain::Triple(toolchain::sys::getProcessTriple()));

  TestContext(toolchain::Triple target)
      : TestContext(DoNotDeclareOptionalTypes, target) {};

  template <typename Nominal>
  typename std::enable_if<!std::is_same<Nominal, language::ClassDecl>::value,
                          Nominal *>::type
  makeNominal(StringRef name, GenericParamList *genericParams = nullptr) {
    auto result = new (Ctx) Nominal(SourceLoc(), Ctx.getIdentifier(name),
                                    SourceLoc(), /*inherited*/{},
                                    genericParams, FileForLookups);
    result->setAccess(AccessLevel::Internal);
    return result;
  }

  template <typename Nominal>
  typename std::enable_if<std::is_same<Nominal, language::ClassDecl>::value,
                          language::ClassDecl *>::type
  makeNominal(StringRef name, GenericParamList *genericParams = nullptr) {
    auto result = new (Ctx) ClassDecl(SourceLoc(), Ctx.getIdentifier(name),
                                      SourceLoc(), /*inherited*/{},
                                      genericParams, FileForLookups,
                                      /*isActor*/false);
    result->setAccess(AccessLevel::Internal);
    return result;
  }

};

} // end namespace unittest
} // end namespace language
