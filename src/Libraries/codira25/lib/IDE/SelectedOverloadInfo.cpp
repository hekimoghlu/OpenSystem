/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

//===--- SelectedOverloadInfo.cpp -----------------------------------------===//
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

#include "language/Basic/Assertions.h"
#include "language/IDE/SelectedOverloadInfo.h"

using namespace language::ide;

SelectedOverloadInfo
language::ide::getSelectedOverloadInfo(const Solution &S,
                                    ConstraintLocator *CalleeLocator) {
  auto &CS = S.getConstraintSystem();

  SelectedOverloadInfo Result;

  auto SelectedOverload = S.getOverloadChoiceIfAvailable(CalleeLocator);
  if (!SelectedOverload) {
    return Result;
  }

  switch (SelectedOverload->choice.getKind()) {
  case OverloadChoiceKind::KeyPathApplication:
  case OverloadChoiceKind::Decl:
  case OverloadChoiceKind::DeclViaDynamic:
  case OverloadChoiceKind::DeclViaBridge:
  case OverloadChoiceKind::DeclViaUnwrappedOptional: {
    Result.BaseTy = SelectedOverload->choice.getBaseType();
    if (Result.BaseTy) {
      Result.BaseTy = S.simplifyType(Result.BaseTy)->getRValueType();
    }
    if (Result.BaseTy && Result.BaseTy->is<ModuleType>()) {
      Result.BaseTy = nullptr;
    }

    if (auto ReferencedDecl = SelectedOverload->choice.getDeclOrNull()) {
      Result.ValueRef = S.resolveConcreteDeclRef(ReferencedDecl, CalleeLocator);
    }
    Result.ValueTy =
        S.simplifyTypeForCodeCompletion(SelectedOverload->adjustedOpenedType);

    // For completion as the arg in a call to the implicit [keypath: _]
    // subscript the solver can't know what kind of keypath is expected without
    // an actual argument (e.g. a KeyPath vs WritableKeyPath) so it ends up as a
    // hole. Just assume KeyPath so we show the expected keypath's root type to
    // users rather than '_'.
    if (SelectedOverload->choice.getKind() ==
        OverloadChoiceKind::KeyPathApplication) {
      auto Params = Result.ValueTy->getAs<AnyFunctionType>()->getParams();
      if (Params.size() == 1 &&
          Params[0].getPlainType()->is<UnresolvedType>()) {
        auto *KPDecl = CS.getASTContext().getKeyPathDecl();
        Type KPTy =
            KPDecl->mapTypeIntoContext(KPDecl->getDeclaredInterfaceType());
        Type KPValueTy = KPTy->castTo<BoundGenericType>()->getGenericArgs()[1];
        KPTy =
            BoundGenericType::get(KPDecl, Type(), {Result.BaseTy, KPValueTy});
        Result.ValueTy =
            FunctionType::get({Params[0].withType(KPTy)}, KPValueTy);
      }
    }
    break;
  }
  case OverloadChoiceKind::KeyPathDynamicMemberLookup: {
    auto *fnType = SelectedOverload->adjustedOpenedType->castTo<FunctionType>();
    assert(fnType->getParams().size() == 1 &&
           "subscript always has one argument");
    // Parameter type is KeyPath<T, U> where `T` is a root type
    // and U is a leaf type (aka member type).
    auto keyPathTy =
        fnType->getParams()[0].getPlainType()->castTo<BoundGenericType>();

    auto *keyPathDecl = keyPathTy->getAnyNominal();
    assert(isKnownKeyPathType(keyPathTy) &&
           "parameter is supposed to be a keypath");

    auto KeyPathDynamicLocator = CS.getConstraintLocator(
        CalleeLocator, LocatorPathElt::KeyPathDynamicMember(keyPathDecl));
    Result = getSelectedOverloadInfo(S, KeyPathDynamicLocator);
    break;
  }
  case OverloadChoiceKind::DynamicMemberLookup:
  case OverloadChoiceKind::TupleIndex:
  case OverloadChoiceKind::MaterializePack:
  case OverloadChoiceKind::ExtractFunctionIsolation:
    // If it's DynamicMemberLookup, we don't know which function is being
    // called, so we can't extract any information from it.
    // TupleIndex isn't a function call and is not relevant for argument
    // completion because it doesn't take arguments.
    break;
  }

  return Result;
}
