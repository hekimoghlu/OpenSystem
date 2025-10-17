/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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

//===--- SemaFixture.h - Helper for setting up Sema context -----*- C++ -*-===//
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
#include "language/AST/Type.h"
#include "language/AST/Types.h"
#include "language/Basic/LangOptions.h"
#include "language/Basic/Platform.h"
#include "language/Basic/SourceManager.h"
#include "language/Sema/ConstraintSystem.h"
#include "language/SymbolGraphGen/SymbolGraphOptions.h"
#include "toolchain/ADT/SmallString.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/TargetParser/Host.h"
#include "toolchain/Support/Path.h"
#include "gtest/gtest.h"
#include <string>

using namespace language::constraints;
using namespace language::constraints::inference;

namespace language {
namespace unittest {

class SemaTestBase : public ::testing::Test {
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

  SemaTestBase() : Diags(SourceMgr) {
    LangOpts.Target = toolchain::Triple(toolchain::sys::getDefaultTargetTriple());

    toolchain::SmallString<128> libDir(LANGUAGELIB_DIR);
    toolchain::sys::path::append(libDir, getPlatformNameForTriple(LangOpts.Target));

    SearchPathOpts.RuntimeResourcePath = LANGUAGELIB_DIR;
    SearchPathOpts.RuntimeLibraryPaths.push_back(std::string(libDir.str()));
    SearchPathOpts.setRuntimeLibraryImportPaths({libDir.str().str()});
  }
};

/// Owns an ASTContext and the associated types.
class SemaTest : public SemaTestBase {
  SourceFile *MainFile;

public:
  ASTContext &Context;
  DeclContext *DC;

  SemaTest();

protected:
  Type getStdlibType(StringRef name) const;

  NominalTypeDecl *getStdlibNominalTypeDecl(StringRef name) const;

  VarDecl *addExtensionVarMember(NominalTypeDecl *decl, StringRef name,
                                 Type type) const;

  ProtocolType *createProtocol(toolchain::StringRef protocolName,
                               Type parent = Type());

  static BindingSet inferBindings(ConstraintSystem &cs,
                                  TypeVariableType *typeVar);
};

} // end namespace unittest
} // end namespace language
