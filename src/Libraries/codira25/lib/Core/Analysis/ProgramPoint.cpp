/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

//==- ProgramPoint.cpp - Program Points for Path-Sensitive Analysis -*- C++ -*-/
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface ProgramPoint, which identifies a
//  distinct location in a function.
//
//===----------------------------------------------------------------------===//

#include "language/Core/Analysis/ProgramPoint.h"
#include "language/Core/AST/ASTContext.h"
#include "language/Core/Analysis/AnalysisDeclContext.h"
#include "language/Core/Basic/JsonSupport.h"

using namespace language::Core;

ProgramPointTag::~ProgramPointTag() {}

ProgramPoint ProgramPoint::getProgramPoint(const Stmt *S, ProgramPoint::Kind K,
                                           const LocationContext *LC,
                                           const ProgramPointTag *tag){
  switch (K) {
    default:
      toolchain_unreachable("Unhandled ProgramPoint kind");
    case ProgramPoint::PreStmtKind:
      return PreStmt(S, LC, tag);
    case ProgramPoint::PostStmtKind:
      return PostStmt(S, LC, tag);
    case ProgramPoint::PreLoadKind:
      return PreLoad(S, LC, tag);
    case ProgramPoint::PostLoadKind:
      return PostLoad(S, LC, tag);
    case ProgramPoint::PreStoreKind:
      return PreStore(S, LC, tag);
    case ProgramPoint::PostLValueKind:
      return PostLValue(S, LC, tag);
    case ProgramPoint::PostStmtPurgeDeadSymbolsKind:
      return PostStmtPurgeDeadSymbols(S, LC, tag);
    case ProgramPoint::PreStmtPurgeDeadSymbolsKind:
      return PreStmtPurgeDeadSymbols(S, LC, tag);
  }
}

LLVM_DUMP_METHOD void ProgramPoint::dump() const {
  return printJson(toolchain::errs());
}

StringRef ProgramPoint::getProgramPointKindName(Kind K) {
  switch (K) {
  case BlockEdgeKind:
    return "BlockEdge";
  case BlockEntranceKind:
    return "BlockEntrance";
  case BlockExitKind:
    return "BlockExit";
  case PreStmtKind:
    return "PreStmt";
  case PreStmtPurgeDeadSymbolsKind:
    return "PreStmtPurgeDeadSymbols";
  case PostStmtPurgeDeadSymbolsKind:
    return "PostStmtPurgeDeadSymbols";
  case PostStmtKind:
    return "PostStmt";
  case PreLoadKind:
    return "PreLoad";
  case PostLoadKind:
    return "PostLoad";
  case PreStoreKind:
    return "PreStore";
  case PostStoreKind:
    return "PostStore";
  case PostConditionKind:
    return "PostCondition";
  case PostLValueKind:
    return "PostLValue";
  case PostAllocatorCallKind:
    return "PostAllocatorCall";
  case PostInitializerKind:
    return "PostInitializer";
  case CallEnterKind:
    return "CallEnter";
  case CallExitBeginKind:
    return "CallExitBegin";
  case CallExitEndKind:
    return "CallExitEnd";
  case FunctionExitKind:
    return "FunctionExit";
  case PreImplicitCallKind:
    return "PreImplicitCall";
  case PostImplicitCallKind:
    return "PostImplicitCall";
  case LoopExitKind:
    return "LoopExit";
  case EpsilonKind:
    return "Epsilon";
  }
  toolchain_unreachable("Unknown ProgramPoint kind");
}

std::optional<SourceLocation> ProgramPoint::getSourceLocation() const {
  switch (getKind()) {
  case BlockEdgeKind:
    // If needed, the source and or destination beginning can be used to get
    // source location.
    return std::nullopt;
  case BlockEntranceKind:
    // If needed, first statement of the block can be used.
    return std::nullopt;
  case BlockExitKind:
    if (const auto *B = castAs<BlockExit>().getBlock()) {
      if (const auto *T = B->getTerminatorStmt()) {
        return T->getBeginLoc();
      }
    }
    return std::nullopt;
  case PreStmtKind:
  case PreStmtPurgeDeadSymbolsKind:
  case PostStmtPurgeDeadSymbolsKind:
  case PostStmtKind:
  case PreLoadKind:
  case PostLoadKind:
  case PreStoreKind:
  case PostStoreKind:
  case PostConditionKind:
  case PostLValueKind:
  case PostAllocatorCallKind:
    if (const Stmt *S = castAs<StmtPoint>().getStmt())
      return S->getBeginLoc();
    return std::nullopt;
  case PostInitializerKind:
    if (const auto *Init = castAs<PostInitializer>().getInitializer())
      return Init->getSourceLocation();
    return std::nullopt;
  case CallEnterKind:
    if (const Stmt *S = castAs<CallEnter>().getCallExpr())
      return S->getBeginLoc();
    return std::nullopt;
  case CallExitBeginKind:
    if (const Stmt *S = castAs<CallExitBegin>().getReturnStmt())
      return S->getBeginLoc();
    return std::nullopt;
  case CallExitEndKind:
    return std::nullopt;
  case FunctionExitKind:
    if (const auto *B = castAs<FunctionExitPoint>().getBlock();
        B && B->getTerminatorStmt())
      return B->getTerminatorStmt()->getBeginLoc();
    return std::nullopt;
  case PreImplicitCallKind:
    return castAs<ImplicitCallPoint>().getLocation();
  case PostImplicitCallKind:
    return castAs<ImplicitCallPoint>().getLocation();
  case LoopExitKind:
    if (const Stmt *S = castAs<LoopExit>().getLoopStmt())
      return S->getBeginLoc();
    return std::nullopt;
  case EpsilonKind:
    return std::nullopt;
  }
  toolchain_unreachable("Unknown ProgramPoint kind");
}

void ProgramPoint::printJson(toolchain::raw_ostream &Out, const char *NL) const {
  const ASTContext &Context =
      getLocationContext()->getAnalysisDeclContext()->getASTContext();
  const SourceManager &SM = Context.getSourceManager();
  const PrintingPolicy &PP = Context.getPrintingPolicy();
  const bool AddQuotes = true;

  Out << "\"kind\": \"";
  switch (getKind()) {
  case ProgramPoint::BlockEntranceKind:
    Out << "BlockEntrance\""
        << ", \"block_id\": "
        << castAs<BlockEntrance>().getBlock()->getBlockID();
    break;

  case ProgramPoint::FunctionExitKind: {
    auto FEP = getAs<FunctionExitPoint>();
    Out << "FunctionExit\""
        << ", \"block_id\": " << FEP->getBlock()->getBlockID()
        << ", \"stmt_id\": ";

    if (const ReturnStmt *RS = FEP->getStmt()) {
      Out << RS->getID(Context) << ", \"stmt\": ";
      RS->printJson(Out, nullptr, PP, AddQuotes);
    } else {
      Out << "null, \"stmt\": null";
    }
    break;
  }
  case ProgramPoint::BlockExitKind:
    toolchain_unreachable("BlockExitKind");
    break;
  case ProgramPoint::CallEnterKind:
    Out << "CallEnter\", \"callee_decl\": \"";
    Out << AnalysisDeclContext::getFunctionName(
               castAs<CallEnter>().getCalleeContext()->getDecl())
        << '\"';
    break;
  case ProgramPoint::CallExitBeginKind:
    Out << "CallExitBegin\"";
    break;
  case ProgramPoint::CallExitEndKind:
    Out << "CallExitEnd\"";
    break;
  case ProgramPoint::EpsilonKind:
    Out << "EpsilonPoint\"";
    break;

  case ProgramPoint::LoopExitKind:
    Out << "LoopExit\", \"stmt\": \""
        << castAs<LoopExit>().getLoopStmt()->getStmtClassName() << '\"';
    break;

  case ProgramPoint::PreImplicitCallKind: {
    ImplicitCallPoint PC = castAs<ImplicitCallPoint>();
    Out << "PreCall\", \"decl\": \""
        << PC.getDecl()->getAsFunction()->getQualifiedNameAsString()
        << "\", \"location\": ";
    printSourceLocationAsJson(Out, PC.getLocation(), SM);
    break;
  }

  case ProgramPoint::PostImplicitCallKind: {
    ImplicitCallPoint PC = castAs<ImplicitCallPoint>();
    Out << "PostCall\", \"decl\": \""
        << PC.getDecl()->getAsFunction()->getQualifiedNameAsString()
        << "\", \"location\": ";
    printSourceLocationAsJson(Out, PC.getLocation(), SM);
    break;
  }

  case ProgramPoint::PostInitializerKind: {
    Out << "PostInitializer\", ";
    const CXXCtorInitializer *Init = castAs<PostInitializer>().getInitializer();
    if (const FieldDecl *FD = Init->getAnyMember()) {
      Out << "\"field_decl\": \"" << *FD << '\"';
    } else {
      Out << "\"type\": \"";
      QualType Ty = Init->getTypeSourceInfo()->getType();
      Ty = Ty.getLocalUnqualifiedType();
      Ty.print(Out, Context.getLangOpts());
      Out << '\"';
    }
    break;
  }

  case ProgramPoint::BlockEdgeKind: {
    const BlockEdge &E = castAs<BlockEdge>();
    const Stmt *T = E.getSrc()->getTerminatorStmt();
    Out << "Edge\", \"src_id\": " << E.getSrc()->getBlockID()
        << ", \"dst_id\": " << E.getDst()->getBlockID() << ", \"terminator\": ";

    if (!T) {
      Out << "null, \"term_kind\": null";
      break;
    }

    E.getSrc()->printTerminatorJson(Out, Context.getLangOpts(),
                                    /*AddQuotes=*/true);
    Out << ", \"location\": ";
    printSourceLocationAsJson(Out, T->getBeginLoc(), SM);

    Out << ", \"term_kind\": \"";
    if (isa<SwitchStmt>(T)) {
      Out << "SwitchStmt\", \"case\": ";
      if (const Stmt *Label = E.getDst()->getLabel()) {
        if (const auto *C = dyn_cast<CaseStmt>(Label)) {
          Out << "{ \"lhs\": ";
          if (const Stmt *LHS = C->getLHS()) {
            LHS->printJson(Out, nullptr, PP, AddQuotes);
          } else {
            Out << "null";
          }

          Out << ", \"rhs\": ";
          if (const Stmt *RHS = C->getRHS()) {
            RHS->printJson(Out, nullptr, PP, AddQuotes);
          } else {
            Out << "null";
          }
          Out << " }";
        } else {
          assert(isa<DefaultStmt>(Label));
          Out << "\"default\"";
        }
      } else {
        Out << "\"implicit default\"";
      }
    } else if (isa<IndirectGotoStmt>(T)) {
      // FIXME: More info.
      Out << "IndirectGotoStmt\"";
    } else {
      Out << "Condition\", \"value\": "
          << (*E.getSrc()->succ_begin() == E.getDst() ? "true" : "false");
    }
    break;
  }

  default: {
    const Stmt *S = castAs<StmtPoint>().getStmt();
    assert(S != nullptr && "Expecting non-null Stmt");

    Out << "Statement\", \"stmt_kind\": \"" << S->getStmtClassName()
        << "\", \"stmt_id\": " << S->getID(Context)
        << ", \"pointer\": \"" << (const void *)S << "\", ";
    if (const auto *CS = dyn_cast<CastExpr>(S))
      Out << "\"cast_kind\": \"" << CS->getCastKindName() << "\", ";

    Out << "\"pretty\": ";

    S->printJson(Out, nullptr, PP, AddQuotes);

    Out << ", \"location\": ";
    printSourceLocationAsJson(Out, S->getBeginLoc(), SM);

    Out << ", \"stmt_point_kind\": \"";
    if (getAs<PreLoad>())
      Out << "PreLoad";
    else if (getAs<PreStore>())
      Out << "PreStore";
    else if (getAs<PostAllocatorCall>())
      Out << "PostAllocatorCall";
    else if (getAs<PostCondition>())
      Out << "PostCondition";
    else if (getAs<PostLoad>())
      Out << "PostLoad";
    else if (getAs<PostLValue>())
      Out << "PostLValue";
    else if (getAs<PostStore>())
      Out << "PostStore";
    else if (getAs<PostStmt>())
      Out << "PostStmt";
    else if (getAs<PostStmtPurgeDeadSymbols>())
      Out << "PostStmtPurgeDeadSymbols";
    else if (getAs<PreStmtPurgeDeadSymbols>())
      Out << "PreStmtPurgeDeadSymbols";
    else if (getAs<PreStmt>())
      Out << "PreStmt";
    else {
      Out << "\nKind: '" << getKind();
      toolchain_unreachable("' is unhandled StmtPoint kind!");
    }

    Out << '\"';
    break;
  }
  }
}

SimpleProgramPointTag::SimpleProgramPointTag(StringRef MsgProvider,
                                             StringRef Msg)
  : Desc((MsgProvider + " : " + Msg).str()) {}

StringRef SimpleProgramPointTag::getDebugTag() const { return Desc; }
