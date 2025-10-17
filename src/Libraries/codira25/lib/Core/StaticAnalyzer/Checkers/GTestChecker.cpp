/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

//==- GTestChecker.cpp - Model gtest API --*- C++ -*-==//
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
// This checker models the behavior of un-inlined APIs from the gtest
// unit-testing library to avoid false positives when using assertions from
// that library.
//
//===----------------------------------------------------------------------===//

#include "language/Core/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "language/Core/AST/Expr.h"
#include "language/Core/Basic/LangOptions.h"
#include "language/Core/StaticAnalyzer/Core/Checker.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "toolchain/Support/raw_ostream.h"
#include <optional>

using namespace language::Core;
using namespace ento;

// Modeling of un-inlined AssertionResult constructors
//
// The gtest unit testing API provides macros for assertions that expand
// into an if statement that calls a series of constructors and returns
// when the "assertion" is false.
//
// For example,
//
//   ASSERT_TRUE(a == b)
//
// expands into:
//
//   switch (0)
//   case 0:
//   default:
//     if (const ::testing::AssertionResult gtest_ar_ =
//             ::testing::AssertionResult((a == b)))
//       ;
//     else
//       return ::testing::internal::AssertHelper(
//                  ::testing::TestPartResult::kFatalFailure,
//                  "<path to project>",
//                  <line number>,
//                  ::testing::internal::GetBoolAssertionFailureMessage(
//                      gtest_ar_, "a == b", "false", "true")
//                      .c_str()) = ::testing::Message();
//
// where AssertionResult is defined similarly to
//
//   class AssertionResult {
//   public:
//     AssertionResult(const AssertionResult& other);
//     explicit AssertionResult(bool success) : success_(success) {}
//     operator bool() const { return success_; }
//     ...
//     private:
//     bool success_;
//   };
//
// In order for the analyzer to correctly handle this assertion, it needs to
// know that the boolean value of the expression "a == b" is stored the
// 'success_' field of the original AssertionResult temporary and propagated
// (via the copy constructor) into the 'success_' field of the object stored
// in 'gtest_ar_'.  That boolean value will then be returned from the bool
// conversion method in the if statement. This guarantees that the assertion
// holds when the return path is not taken.
//
// If the success value is not properly propagated, then the eager case split
// on evaluating the expression can cause pernicious false positives
// on the non-return path:
//
//   ASSERT(ptr != NULL)
//   *ptr = 7; // False positive null pointer dereference here
//
// Unfortunately, the bool constructor cannot be inlined (because its
// implementation is not present in the headers) and the copy constructor is
// not inlined (because it is constructed into a temporary and the analyzer
// does not inline these since it does not yet reliably call temporary
// destructors).
//
// This checker compensates for the missing inlining by propagating the
// _success value across the bool and copy constructors so the assertion behaves
// as expected.

namespace {
class GTestChecker : public Checker<check::PostCall> {

  mutable IdentifierInfo *AssertionResultII = nullptr;
  mutable IdentifierInfo *SuccessII = nullptr;

public:
  GTestChecker() = default;

  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

private:
  void modelAssertionResultBoolConstructor(const CXXConstructorCall *Call,
                                           bool IsRef, CheckerContext &C) const;

  void modelAssertionResultCopyConstructor(const CXXConstructorCall *Call,
                                           CheckerContext &C) const;

  void initIdentifierInfo(ASTContext &Ctx) const;

  SVal
  getAssertionResultSuccessFieldValue(const CXXRecordDecl *AssertionResultDecl,
                                      SVal Instance,
                                      ProgramStateRef State) const;

  static ProgramStateRef assumeValuesEqual(SVal Val1, SVal Val2,
                                           ProgramStateRef State,
                                           CheckerContext &C);
};
} // End anonymous namespace.

/// Model a call to an un-inlined AssertionResult(bool) or
/// AssertionResult(bool &, ...).
/// To do so, constrain the value of the newly-constructed instance's 'success_'
/// field to be equal to the passed-in boolean value.
///
/// \param IsRef Whether the boolean parameter is a reference or not.
void GTestChecker::modelAssertionResultBoolConstructor(
    const CXXConstructorCall *Call, bool IsRef, CheckerContext &C) const {
  assert(Call->getNumArgs() >= 1 && Call->getNumArgs() <= 2);

  ProgramStateRef State = C.getState();
  SVal BooleanArgVal = Call->getArgSVal(0);
  if (IsRef) {
    // The argument is a reference, so load from it to get the boolean value.
    if (!isa<Loc>(BooleanArgVal))
      return;
    BooleanArgVal = C.getState()->getSVal(BooleanArgVal.castAs<Loc>());
  }

  SVal ThisVal = Call->getCXXThisVal();

  SVal ThisSuccess = getAssertionResultSuccessFieldValue(
      Call->getDecl()->getParent(), ThisVal, State);

  State = assumeValuesEqual(ThisSuccess, BooleanArgVal, State, C);
  C.addTransition(State);
}

/// Model a call to an un-inlined AssertionResult copy constructor:
///
///   AssertionResult(const &AssertionResult other)
///
/// To do so, constrain the value of the newly-constructed instance's
/// 'success_' field to be equal to the value of the pass-in instance's
/// 'success_' field.
void GTestChecker::modelAssertionResultCopyConstructor(
    const CXXConstructorCall *Call, CheckerContext &C) const {
  assert(Call->getNumArgs() == 1);

  // The first parameter of the copy constructor must be the other
  // instance to initialize this instances fields from.
  SVal OtherVal = Call->getArgSVal(0);
  SVal ThisVal = Call->getCXXThisVal();

  const CXXRecordDecl *AssertResultClassDecl = Call->getDecl()->getParent();
  ProgramStateRef State = C.getState();

  SVal ThisSuccess = getAssertionResultSuccessFieldValue(AssertResultClassDecl,
                                                         ThisVal, State);
  SVal OtherSuccess = getAssertionResultSuccessFieldValue(AssertResultClassDecl,
                                                          OtherVal, State);

  State = assumeValuesEqual(ThisSuccess, OtherSuccess, State, C);
  C.addTransition(State);
}

/// Model calls to AssertionResult constructors that are not inlined.
void GTestChecker::checkPostCall(const CallEvent &Call,
                                 CheckerContext &C) const {
  /// If the constructor was inlined, there is no need model it.
  if (C.wasInlined)
    return;

  initIdentifierInfo(C.getASTContext());

  auto *CtorCall = dyn_cast<CXXConstructorCall>(&Call);
  if (!CtorCall)
    return;

  const CXXConstructorDecl *CtorDecl = CtorCall->getDecl();
  const CXXRecordDecl *CtorParent = CtorDecl->getParent();
  if (CtorParent->getIdentifier() != AssertionResultII)
    return;

  unsigned ParamCount = CtorDecl->getNumParams();

  // Call the appropriate modeling method based the parameters and their
  // types.

  // We have AssertionResult(const &AssertionResult)
  if (CtorDecl->isCopyConstructor() && ParamCount == 1) {
    modelAssertionResultCopyConstructor(CtorCall, C);
    return;
  }

  // There are two possible boolean constructors, depending on which
  // version of gtest is being used:
  //
  // v1.7 and earlier:
  //      AssertionResult(bool success)
  //
  // v1.8 and greater:
  //      template <typename T>
  //      AssertionResult(const T& success,
  //                      typename internal::EnableIf<
  //                          !internal::ImplicitlyConvertible<T,
  //                              AssertionResult>::value>::type*)
  //
  CanQualType BoolTy = C.getASTContext().BoolTy;
  if (ParamCount == 1 && CtorDecl->getParamDecl(0)->getType() == BoolTy) {
    // We have AssertionResult(bool)
    modelAssertionResultBoolConstructor(CtorCall, /*IsRef=*/false, C);
    return;
  }
  if (ParamCount == 2){
    auto *RefTy = CtorDecl->getParamDecl(0)->getType()->getAs<ReferenceType>();
    if (RefTy &&
        RefTy->getPointeeType()->getCanonicalTypeUnqualified() == BoolTy) {
      // We have AssertionResult(bool &, ...)
      modelAssertionResultBoolConstructor(CtorCall, /*IsRef=*/true, C);
      return;
    }
  }
}

void GTestChecker::initIdentifierInfo(ASTContext &Ctx) const {
  if (AssertionResultII)
    return;

  AssertionResultII = &Ctx.Idents.get("AssertionResult");
  SuccessII = &Ctx.Idents.get("success_");
}

/// Returns the value stored in the 'success_' field of the passed-in
/// AssertionResult instance.
SVal GTestChecker::getAssertionResultSuccessFieldValue(
    const CXXRecordDecl *AssertionResultDecl, SVal Instance,
    ProgramStateRef State) const {

  DeclContext::lookup_result Result = AssertionResultDecl->lookup(SuccessII);
  if (Result.empty())
    return UnknownVal();

  auto *SuccessField = dyn_cast<FieldDecl>(Result.front());
  if (!SuccessField)
    return UnknownVal();

  std::optional<Loc> FieldLoc =
      State->getLValue(SuccessField, Instance).getAs<Loc>();
  if (!FieldLoc)
    return UnknownVal();

  return State->getSVal(*FieldLoc);
}

/// Constrain the passed-in state to assume two values are equal.
ProgramStateRef GTestChecker::assumeValuesEqual(SVal Val1, SVal Val2,
                                                ProgramStateRef State,
                                                CheckerContext &C) {
  auto DVal1 = Val1.getAs<DefinedOrUnknownSVal>();
  auto DVal2 = Val2.getAs<DefinedOrUnknownSVal>();
  if (!DVal1 || !DVal2)
    return State;

  auto ValuesEqual =
      C.getSValBuilder().evalEQ(State, *DVal1, *DVal2).getAs<DefinedSVal>();
  if (!ValuesEqual)
    return State;

  State = C.getConstraintManager().assume(State, *ValuesEqual, true);
  return State;
}

void ento::registerGTestChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<GTestChecker>();
}

bool ento::shouldRegisterGTestChecker(const CheckerManager &mgr) {
  // gtest is a C++ API so there is no sense running the checker
  // if not compiling for C++.
  const LangOptions &LO = mgr.getLangOpts();
  return LO.CPlusPlus;
}
