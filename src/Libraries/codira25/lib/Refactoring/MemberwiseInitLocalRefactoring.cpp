/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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

//===----------------------------------------------------------------------===//
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

#include "RefactoringActions.h"

using namespace language::refactoring;

namespace {
struct MemberwiseParameter {
  CharSourceRange NameRange;
  Type MemberType;
  Expr *DefaultExpr;

  MemberwiseParameter(CharSourceRange nameRange, Type type, Expr *initialExpr)
      : NameRange(nameRange), MemberType(type), DefaultExpr(initialExpr) {}
};
} // namespace

static void generateMemberwiseInit(SourceEditConsumer &EditConsumer,
                                   SourceManager &SM,
                                   ArrayRef<MemberwiseParameter> memberVector,
                                   SourceLoc targetLocation) {

  EditConsumer.accept(SM, targetLocation, "\ninternal init(");
  auto insertMember = [&SM](const MemberwiseParameter &memberData,
                            raw_ostream &OS, bool wantsSeparator) {
    {
      OS << SM.extractText(memberData.NameRange) << ": ";
      // Unconditionally print '@escaping' if we print out a function type -
      // the assignments we generate below will escape this parameter.
      if (isa<AnyFunctionType>(memberData.MemberType->getCanonicalType())) {
        OS << "@" << TypeAttribute::getAttrName(TypeAttrKind::Escaping) << " ";
      }
      OS << memberData.MemberType.getString();
    }

    bool HasAddedDefault = false;
    if (auto *expr = memberData.DefaultExpr) {
      if (expr->getSourceRange().isValid()) {
        auto range = Lexer::getCharSourceRangeFromSourceRange(
            SM, expr->getSourceRange());
        OS << " = " << SM.extractText(range);
        HasAddedDefault = true;
      }
    }
    if (!HasAddedDefault && memberData.MemberType->isOptional()) {
      OS << " = nil";
    }

    if (wantsSeparator) {
      OS << ", ";
    }
  };

  // Process the initial list of members, inserting commas as appropriate.
  std::string Buffer;
  toolchain::raw_string_ostream OS(Buffer);
  for (const auto &memberData : toolchain::enumerate(memberVector)) {
    bool wantsSeparator = (memberData.index() != memberVector.size() - 1);
    insertMember(memberData.value(), OS, wantsSeparator);
  }

  // Synthesize the body.
  OS << ") {\n";
  for (auto &member : memberVector) {
    // self.<property> = <property>
    auto name = SM.extractText(member.NameRange);
    OS << "self." << name << " = " << name << "\n";
  }
  OS << "}\n";

  // Accept the entire edit.
  EditConsumer.accept(SM, targetLocation, OS.str());
}

static SourceLoc
collectMembersForInit(ResolvedCursorInfoPtr CursorInfo,
                      SmallVectorImpl<MemberwiseParameter> &memberVector) {
  auto ValueRefInfo = dyn_cast<ResolvedValueRefCursorInfo>(CursorInfo);
  if (!ValueRefInfo || !ValueRefInfo->getValueD())
    return SourceLoc();

  NominalTypeDecl *nominalDecl =
      dyn_cast<NominalTypeDecl>(ValueRefInfo->getValueD());
  if (!nominalDecl || nominalDecl->getStoredProperties().empty() ||
      ValueRefInfo->isRef()) {
    return SourceLoc();
  }

  SourceLoc bracesStart = nominalDecl->getBraces().Start;
  if (!bracesStart.isValid())
    return SourceLoc();

  SourceLoc targetLocation = bracesStart.getAdvancedLoc(1);
  if (!targetLocation.isValid())
    return SourceLoc();

  SourceManager &SM = nominalDecl->getASTContext().SourceMgr;

  for (auto member : nominalDecl->getMemberwiseInitProperties()) {
    auto varDecl = dyn_cast<VarDecl>(member);
    if (!varDecl) {
      continue;
    }
    if (varDecl->getAttrs().hasAttribute<LazyAttr>()) {
      // Exclude lazy members from the memberwise initializer. This is
      // inconsistent with the implicitly synthesized memberwise initializer but
      // we think it makes more sense because otherwise the lazy variable's
      // initializer gets evaluated eagerly.
      continue;
    }

    auto patternBinding = varDecl->getParentPatternBinding();
    if (!patternBinding)
      continue;

    const auto i = patternBinding->getPatternEntryIndexForVarDecl(varDecl);
    Expr *defaultInit = nullptr;
    if (patternBinding->isExplicitlyInitialized(i) ||
        patternBinding->isDefaultInitializable()) {
      defaultInit = patternBinding->getOriginalInit(i);
    }

    auto NameRange =
        Lexer::getCharSourceRangeFromSourceRange(SM, varDecl->getNameLoc());
    memberVector.emplace_back(NameRange, varDecl->getTypeInContext(),
                              defaultInit);
  }

  return targetLocation;
}

bool RefactoringActionMemberwiseInitLocalRefactoring::isApplicable(
    ResolvedCursorInfoPtr Tok, DiagnosticEngine &Diag) {

  SmallVector<MemberwiseParameter, 8> memberVector;
  return collectMembersForInit(Tok, memberVector).isValid();
}

bool RefactoringActionMemberwiseInitLocalRefactoring::performChange() {

  SmallVector<MemberwiseParameter, 8> memberVector;
  SourceLoc targetLocation = collectMembersForInit(CursorInfo, memberVector);
  if (targetLocation.isInvalid())
    return true;

  generateMemberwiseInit(EditConsumer, SM, memberVector, targetLocation);

  return false;
}
