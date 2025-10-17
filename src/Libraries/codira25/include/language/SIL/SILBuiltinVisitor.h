/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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

//===--- SILBuiltinVisitor.h ------------------------------------*- C++ -*-===//
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
///
/// \file
///
/// This file contains SILBuiltinVisitor, a visitor for visiting all possible
/// builtins and toolchain intrinsics able to be used by BuiltinInst.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SIL_SILBUILTINVISITOR_H
#define LANGUAGE_SIL_SILBUILTINVISITOR_H

#include "language/SIL/SILInstruction.h"
#include <type_traits>

namespace language {

template <typename ImplClass, typename ValueRetTy = void>
class SILBuiltinVisitor {
public:
  ImplClass &asImpl() { return static_cast<ImplClass &>(*this); }

  /// Perform any required pre-processing before visiting.
  ///
  /// Sub-classes can override this method to provide custom pre-processing.
  void beforeVisit(BuiltinInst *BI) {}

  ValueRetTy visit(BuiltinInst *BI) {
    asImpl().beforeVisit(BI);

    if (auto BuiltinKind = BI->getBuiltinKind()) {
      switch (BuiltinKind.value()) {
      // BUILTIN_TYPE_CHECKER_OPERATION does not live past the type checker.
#define BUILTIN_TYPE_CHECKER_OPERATION(ID, NAME)                               \
  case BuiltinValueKind::ID:                                                   \
    toolchain_unreachable("Unexpected type checker operation seen in SIL!");

#define BUILTIN(ID, NAME, ATTRS)                                               \
  case BuiltinValueKind::ID:                                                   \
    return asImpl().visit##ID(BI, ATTRS);
#include "language/AST/Builtins.def"
      case BuiltinValueKind::None:
        toolchain_unreachable("None case");
      }
      toolchain_unreachable("Not all cases handled?!");
    }

    if (auto IntrinsicID = BI->getIntrinsicID()) {
      return asImpl().visitLLVMIntrinsic(BI, IntrinsicID.value());
    }
    toolchain_unreachable("Not all cases handled?!");
  }

  ValueRetTy visitLLVMIntrinsic(BuiltinInst *BI, toolchain::Intrinsic::ID ID) {
    return ValueRetTy();
  }

  ValueRetTy visitBuiltinValueKind(BuiltinInst *BI, BuiltinValueKind Kind,
                                   StringRef Attrs) {
    return ValueRetTy();
  }

#define BUILTIN(ID, NAME, ATTRS)                                               \
  ValueRetTy visit##ID(BuiltinInst *BI, StringRef) {                           \
    return asImpl().visitBuiltinValueKind(BI, BuiltinValueKind::ID, ATTRS);    \
  }
#include "language/AST/Builtins.def"
};

} // end language namespace

#endif
