/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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

//===--- UnresolvedMemberLookupTests.cpp --------------------------------===//
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

#include "SemaFixture.h"
#include "language/AST/GenericParamList.h"
#include "language/Sema/ConstraintSystem.h"

using namespace language;
using namespace language::unittest;
using namespace language::constraints;

/// Even in the face of a more permissive conversion that might be chosen based
/// on other ranking rules (e.g., the overload required is non-generic), ensure
/// that we will select the solution which requires the more narrow conversion
/// (e.g., the overload *is* generic).
TEST_F(SemaTest, TestKeypathFunctionConversionPrefersNarrowConversion) {
  auto boolType = getStdlibType("Bool");
  auto boolOptType = OptionalType::get(boolType);
  auto stringType = getStdlibType("String");

  auto *genericParam1 = GenericTypeParamDecl::createImplicit(
      DC, Context.getIdentifier("T"), 0, 0, GenericTypeParamKind::Type);
  auto genericType1 =
      genericParam1->getDeclaredInterfaceType()->getAs<GenericTypeParamType>();

  auto *genericParam2 = GenericTypeParamDecl::createImplicit(
      DC, Context.getIdentifier("T"), 0, 1, GenericTypeParamKind::Type);
  auto genericType2 =
      genericParam2->getDeclaredInterfaceType()->getAs<GenericTypeParamType>();

  auto declName = DeclName(Context, Context.getIdentifier("f"), {Identifier()});

  // fn f<T, U>(_: (T) -> U))
  auto innerGenericFnParam = AnyFunctionType::Param(genericType1);
  auto genericFnParamTy = FunctionType::get({innerGenericFnParam}, genericType2)
                              ->withExtInfo(AnyFunctionType::ExtInfo());
  auto *genericFnParamDecl = ParamDecl::createImplicit(
      Context, Identifier(), Identifier(), genericFnParamTy, DC);
  genericFnParamDecl->setSpecifier(ParamSpecifier::Default);
  auto *genericFnParamList =
      ParameterList::createWithoutLoc(genericFnParamDecl);
  toolchain::SmallVector<GenericTypeParamDecl *, 2> genericParams = {genericParam1,
                                                                genericParam2};
  auto *genericParamList =
      GenericParamList::create(Context, SourceLoc(), {}, SourceLoc());
  auto genericFnDecl = FuncDecl::create(
      Context, SourceLoc(), StaticSpellingKind::None, SourceLoc(), declName,
      SourceLoc(), /*async=*/false, SourceLoc(), /*throws=*/false, SourceLoc(),
      nullptr, genericParamList, genericFnParamList, nullptr, DC);
  auto genericFnParam = AnyFunctionType::Param(genericFnParamTy);
  toolchain::SmallVector<GenericTypeParamType *, 2> genericTypeParams = {
      genericType1, genericType2};
  auto genericSig = GenericSignature::get(genericTypeParams, {});
  auto genericFnTy = GenericFunctionType::get(genericSig, {genericFnParam},
                                              Context.TheEmptyTupleType)
                         ->withExtInfo(AnyFunctionType::ExtInfo());
  genericFnDecl->setInterfaceType(genericFnTy);

  // fn f(_: (String) -> Bool?)
  auto innerConcreteFnParam = AnyFunctionType::Param(stringType);
  auto concreteFnParamTy =
      FunctionType::get({innerConcreteFnParam}, boolOptType)
          ->withExtInfo(AnyFunctionType::ExtInfo());
  auto *concreteFnParamDecl = ParamDecl::createImplicit(
      Context, Identifier(), Identifier(), concreteFnParamTy, DC);
  concreteFnParamDecl->setSpecifier(ParamSpecifier::Default);
  auto *concreteFnParamList =
      ParameterList::createWithoutLoc(concreteFnParamDecl);
  auto concreteFnDecl = FuncDecl::create(
      Context, SourceLoc(), StaticSpellingKind::None, SourceLoc(), declName,
      SourceLoc(), /*async=*/false, SourceLoc(), /*throws=*/false, SourceLoc(),
      nullptr, nullptr, concreteFnParamList, nullptr, DC);
  auto concreteFnParam = AnyFunctionType::Param(concreteFnParamTy);
  auto concreteFnTy =
      FunctionType::get({concreteFnParam}, Context.TheEmptyTupleType)
          ->withExtInfo(AnyFunctionType::ExtInfo());
  concreteFnDecl->setInterfaceType(concreteFnTy);

  // \String.isEmpty
  auto *stringDRE = TypeExpr::createImplicitForDecl(
      DeclNameLoc(), stringType->getAnyNominal(), Context.getStdlibModule(),
      stringType);
  auto *isEmptyDE = new (Context) UnresolvedDotExpr(
      stringDRE, SourceLoc(), DeclNameRef(Context.getIdentifier("isEmpty")),
      DeclNameLoc(), false);
  auto *kpExpr = KeyPathExpr::createParsed(Context, SourceLoc(), isEmptyDE,
                                           nullptr, false);

  // f(\String.isEmpty)
  auto kpArg = Argument(SourceLoc(), Identifier(), kpExpr);
  auto *argList = ArgumentList::create(Context, SourceLoc(), {kpArg},
                                       SourceLoc(), std::nullopt, false);
  toolchain::SmallVector<ValueDecl *, 2> fDecls = {genericFnDecl, concreteFnDecl};
  auto *fDRE = new (Context) OverloadedDeclRefExpr(
      fDecls, DeclNameLoc(), FunctionRefInfo::singleBaseNameApply(), false);
  auto *callExpr = CallExpr::create(Context, fDRE, argList, false);

  ConstraintSystem cs(DC, ConstraintSystemOptions());
  auto target = SyntacticElementTarget(callExpr, DC, CTP_Unused, Type(),
                                       /*isDiscarded*/ true);
  ASSERT_FALSE(cs.preCheckTarget(target));
  ASSERT_FALSE(cs.generateConstraints(target));

  SmallVector<Solution, 2> solutions;
  cs.solve(solutions);

  // We should have a solution.
  ASSERT_EQ(solutions.size(), 1u);

  auto &solution = solutions[0];
  auto *locator = cs.getConstraintLocator(fDRE);
  auto choice = solution.getOverloadChoice(locator).choice;

  // We should select the generic function since it requires 'less' conversion.
  ASSERT_EQ(choice.getDecl(), genericFnDecl);
}
