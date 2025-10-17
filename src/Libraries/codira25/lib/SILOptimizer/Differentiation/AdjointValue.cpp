/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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

//===--- AdjointValue.h - Helper class for differentiation ----*- C++ -*---===//
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
// AdjointValue - a symbolic representation for adjoint values enabling
// efficient differentiation by avoiding zero materialization.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "differentiation"

#include "language/SILOptimizer/Differentiation/AdjointValue.h"

void language::autodiff::AdjointValue::print(toolchain::raw_ostream &s) const {
  switch (getKind()) {
  case AdjointValueKind::Zero:
    s << "Zero[" << getType() << ']';
    break;
  case AdjointValueKind::Aggregate:
    s << "Aggregate[" << getType() << "](";
    if (auto *decl = getType().getASTType()->getStructOrBoundGenericStruct()) {
      interleave(
          toolchain::zip(decl->getStoredProperties(), getAggregateElements()),
          [&s](std::tuple<VarDecl *, const AdjointValue &> elt) {
            s << std::get<0>(elt)->getName() << ": ";
            std::get<1>(elt).print(s);
          },
          [&s] { s << ", "; });
    } else if (getType().is<TupleType>()) {
      interleave(
          getAggregateElements(),
          [&s](const AdjointValue &elt) { elt.print(s); }, [&s] { s << ", "; });
    } else {
      toolchain_unreachable("Invalid aggregate");
    }
    s << ')';
    break;
  case AdjointValueKind::Concrete:
    s << "Concrete[" << getType() << "](" << base->value.concrete << ')';
    break;
  case AdjointValueKind::AddElement:
    auto *addElementValue = getAddElementValue();
    auto baseAdjoint = addElementValue->baseAdjoint;
    auto eltToAdd = addElementValue->eltToAdd;

    s << "AddElement[";
    baseAdjoint.print(s);

    s << ", Field(";
    if (addElementValue->isTupleAdjoint()) {
      s << addElementValue->getFieldIndex();
    } else {
      s << addElementValue->getFieldDecl()->getNameStr();
    }
    s << "), ";

    eltToAdd.print(s);

    s << "]";
    break;
  }
}
