/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

//===-- ComponentPath.cpp -------------------------------------------------===//
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

#include "language/Compability/Lower/ComponentPath.h"

static std::function<
    language::Compability::lower::IterationSpace(const language::Compability::lower::IterationSpace &)>
getIdentityFunc() {
  return [](const language::Compability::lower::IterationSpace &s) { return s; };
}

static std::function<
    language::Compability::lower::IterationSpace(const language::Compability::lower::IterationSpace &)>
getNullaryFunc() {
  return [](const language::Compability::lower::IterationSpace &s) {
    language::Compability::lower::IterationSpace newIters(s);
    newIters.clearIndices();
    return newIters;
  };
}

void language::Compability::lower::ComponentPath::clear() {
  reversePath.clear();
  substring = nullptr;
  applied = false;
  prefixComponents.clear();
  trips.clear();
  suffixComponents.clear();
  pc = getIdentityFunc();
}

bool language::Compability::lower::isRankedArrayAccess(const language::Compability::evaluate::ArrayRef &x) {
  for (const language::Compability::evaluate::Subscript &sub : x.subscript()) {
    if (language::Compability::common::visit(
            language::Compability::common::visitors{
                [&](const language::Compability::evaluate::Triplet &) { return true; },
                [&](const language::Compability::evaluate::IndirectSubscriptIntegerExpr &e) {
                  return e.value().Rank() > 0;
                }},
            sub.u))
      return true;
  }
  return false;
}

void language::Compability::lower::ComponentPath::resetPC() { pc = getIdentityFunc(); }

void language::Compability::lower::ComponentPath::setPC(bool isImplicit) {
  pc = isImplicit ? getIdentityFunc() : getNullaryFunc();
  resetExtendCoorRef();
}

language::Compability::lower::ComponentPath::ExtendRefFunc
language::Compability::lower::ComponentPath::getExtendCoorRef() const {
  return hasExtendCoorRef() ? *extendCoorRef : [](mlir::Value v) { return v; };
}
