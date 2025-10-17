/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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

//===--- GenDecl.h - Codira IR generation for some decl ----------*- C++ -*-===//
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
//  This file provides the private interface to some decl emission code.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENDECL_H
#define LANGUAGE_IRGEN_GENDECL_H

#include "DebugTypeInfo.h"
#include "IRGen.h"
#include "language/Basic/OptimizationMode.h"
#include "language/SIL/SILLocation.h"
#include "language/Core/AST/DeclCXX.h"
#include "toolchain/IR/CallingConv.h"
#include "toolchain/Support/CommandLine.h"

namespace toolchain {
  class AttributeList;
  class Function;
  class FunctionType;
  class CallBase;
}
namespace language {
namespace irgen {
  class IRGenModule;
  class LinkEntity;
  class LinkInfo;
  class Signature;

  void updateLinkageForDefinition(IRGenModule &IGM,
                                  toolchain::GlobalValue *global,
                                  const LinkEntity &entity);

  toolchain::Function *createFunction(
      IRGenModule &IGM, LinkInfo &linkInfo, const Signature &signature,
      toolchain::Function *insertBefore = nullptr,
      OptimizationMode FuncOptMode = OptimizationMode::NotSet,
      StackProtectorMode stackProtect = StackProtectorMode::NoStackProtector);

  toolchain::GlobalVariable *
  createVariable(IRGenModule &IGM, LinkInfo &linkInfo, toolchain::Type *objectType,
                 Alignment alignment, DebugTypeInfo DebugType = DebugTypeInfo(),
                 std::optional<SILLocation> DebugLoc = std::nullopt,
                 StringRef DebugName = StringRef());

  toolchain::GlobalVariable *
  createLinkerDirectiveVariable(IRGenModule &IGM, StringRef Name);

  void disableAddressSanitizer(IRGenModule &IGM, toolchain::GlobalVariable *var);

  /// If the calling convention for `ctor` doesn't match the calling convention
  /// that we assumed for it when we imported it as `initializer`, emit and
  /// return a thunk that conforms to the assumed calling convention. The thunk
  /// is marked `alwaysinline`, so it doesn't generate any runtime overhead.
  /// If the assumed calling convention was correct, just return `ctor`.
  ///
  /// See also comments in CXXMethodConventions in SIL/IR/SILFunctionType.cpp.
  toolchain::Constant *
  emitCXXConstructorThunkIfNeeded(IRGenModule &IGM, Signature signature,
                                  const language::Core::CXXConstructorDecl *ctor,
                                  StringRef name, toolchain::Constant *ctorAddress);

  toolchain::CallBase *emitCXXConstructorCall(IRGenFunction &IGF,
                                         const language::Core::CXXConstructorDecl *ctor,
                                         toolchain::FunctionType *ctorFnType,
                                         toolchain::Constant *ctorAddress,
                                         toolchain::ArrayRef<toolchain::Value *> args);

  bool hasValidSignatureForEmbedded(SILFunction *f);
}
}

extern toolchain::cl::opt<bool> UseBasicDynamicReplacement;

#endif
