/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

//===--- SILGenTopLevel.h - Top-level Code Emission -------------*- C++ -*-===//
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

#ifndef LANGUAGE_SILGEN_SILGENTOPLEVEL_H
#define LANGUAGE_SILGEN_SILGENTOPLEVEL_H

#include "SILGen.h"
#include "language/AST/ASTVisitor.h"
#include "language/AST/SourceFile.h"
#include "language/AST/TypeMemberVisitor.h"

namespace language {

namespace Lowering {

/// Generates a `SILFunction` for `TopLevelCodeDecl`s within a
/// source file ran in script mode.
class SILGenTopLevel : public ASTVisitor<SILGenTopLevel> {
public:
  /// Generate SIL for toplevel code into `SGF`
  SILGenTopLevel(SILGenFunction &SGF);

  void visitSourceFile(SourceFile *SF);
  void visitDecl(Decl *D) {}
  void visitNominalTypeDecl(NominalTypeDecl *NTD);
  void visitExtensionDecl(ExtensionDecl *ED);
  void visitAbstractFunctionDecl(AbstractFunctionDecl *AFD);
  void visitAbstractStorageDecl(AbstractStorageDecl *ASD);
  void visitTopLevelCodeDecl(TopLevelCodeDecl *TD);

private:
  /// The `SILGenFunction` where toplevel code is emitted
  SILGenFunction &SGF;

  /// Walks type declarations to scan for instances where unitialized global
  /// variables are captured by function declarations and emits
  /// `mark_function_escape` SIL instructions for these escape points as needed
  class TypeVisitor : public TypeMemberVisitor<TypeVisitor> {
  public:
    /// Emit `mark_function_escape` SIL instructions into `SGF` for encountered
    /// escape points.
    TypeVisitor(SILGenFunction &SGF);
    void visit(Decl *D);
    void visitDecl(Decl *D) {}
    void emit(IterableDeclContext *Ctx);
    virtual void visitPatternBindingDecl(PatternBindingDecl *PD);
    void visitNominalTypeDecl(NominalTypeDecl *ntd);
    void visitAbstractFunctionDecl(AbstractFunctionDecl *AFD);
    void visitAbstractStorageDecl(AbstractStorageDecl *ASD);
    virtual ~TypeVisitor() {}

  private:
    SILGenFunction &SGF;
  };
  class ExtensionVisitor : public TypeVisitor {
  public:
    /// Emit `mark_function_escape` SIL instructions into `SGF` for encountered
    /// escape points.
    ExtensionVisitor(SILGenFunction &SGF);
    void visitPatternBindingDecl(PatternBindingDecl *PD) override;
  };
};

} // namespace Lowering

} // namespace language

#endif
