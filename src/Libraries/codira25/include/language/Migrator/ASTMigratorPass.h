/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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

//===--- ASTMigratorPass.h --------------------------------------*- C++ -*-===//
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
//
// A base class for a syntactic migrator pass that uses the temporary
// language::migrator::EditorAdapter infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_MIGRATOR_ASTMIGRATORPASS_H
#define LANGUAGE_MIGRATOR_ASTMIGRATORPASS_H

#include "language/AST/ASTContext.h"
#include "language/AST/SourceFile.h"
#include "language/Migrator/EditorAdapter.h"

namespace language {
class SourceManager;
struct MigratorOptions;
class DiagnosticEngine;

namespace migrator {
class ASTMigratorPass {
protected:
  EditorAdapter &Editor;
  SourceFile *SF;
  const MigratorOptions &Opts;
  const StringRef Filename;
  const unsigned BufferID;
  SourceManager &SM;
  DiagnosticEngine &Diags;

  ASTMigratorPass(EditorAdapter &Editor, SourceFile *SF,
                  const MigratorOptions &Opts)
    : Editor(Editor), SF(SF), Opts(Opts), Filename(SF->getFilename()),
      BufferID(SF->getBufferID()),
      SM(SF->getASTContext().SourceMgr), Diags(SF->getASTContext().Diags) {}
};

/// Run a general pass to migrate code based on SDK differences in the previous
/// release.
void runAPIDiffMigratorPass(EditorAdapter &Editor,
                            SourceFile *SF,
                            const MigratorOptions &Opts);

/// Run a pass to fix up the new type of 'try?' in Codira 4
void runOptionalTryMigratorPass(EditorAdapter &Editor,
                                SourceFile *SF,
                                const MigratorOptions &Opts);
  
  
} // end namespace migrator
} // end namespace language

#endif // LANGUAGE_MIGRATOR_ASTMIGRATORPASS_H
