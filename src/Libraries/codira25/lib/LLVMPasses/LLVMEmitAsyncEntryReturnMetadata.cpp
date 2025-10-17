/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

//===--- LLVMEmitAsyncEntryReturnMetadata.cpp - Async function metadata ---===//
//

#include "language/LLVMPasses/Passes.h"
#include "toolchain/Pass.h"
#include "toolchain/IR/Constants.h"
#include "toolchain/IR/Module.h"
#include "toolchain/Transforms/Utils/ModuleUtils.h"

using namespace toolchain;
using namespace language;

#define DEBUG_TYPE "language-async-return"

PreservedAnalyses AsyncEntryReturnMetadataPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  bool changed = false;

  SmallVector<toolchain::Function *, 16> asyncEntries;
  SmallVector<toolchain::Function *, 16> asyncReturns;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    if (F.hasFnAttribute("async_entry"))
      asyncEntries.push_back(&F);
    if (F.hasFnAttribute("async_ret"))
      asyncReturns.push_back(&F);
  }

  auto &ctxt = M.getContext();
  auto int32Ty = toolchain::Type::getInt32Ty(ctxt);
  auto sizeTy = M.getDataLayout().getIntPtrType(ctxt, /*addrspace*/ 0);

  auto addSection = [&] (const char * sectionName, const char *globalName,
                         SmallVectorImpl<toolchain::Function *> & entries) {
    if (entries.empty())
      return;

    auto intArrayTy = toolchain::ArrayType::get(int32Ty, entries.size());
    auto global =
      new toolchain::GlobalVariable(M, intArrayTy, true,
                               toolchain::GlobalValue::InternalLinkage,
                               nullptr, /*init*/ globalName,
                               nullptr, /*insertBefore*/
                               toolchain::GlobalValue::NotThreadLocal,
                               0/*address space*/);
    global->setAlignment(Align(4));
    global->setSection(sectionName);
    size_t index = 0;
    SmallVector<toolchain::Constant*, 16> offsets;
    for (auto *fn : entries) {
      toolchain::Constant *indices[] = { toolchain::ConstantInt::get(int32Ty, 0),
        toolchain::ConstantInt::get(int32Ty, index)};
      ++index;

      toolchain::Constant *base = toolchain::ConstantExpr::getInBoundsGetElementPtr(
       intArrayTy, global, indices);
      base = toolchain::ConstantExpr::getPtrToInt(base, sizeTy);
      auto *target = toolchain::ConstantExpr::getPtrToInt(fn, sizeTy);
      toolchain::Constant *offset = toolchain::ConstantExpr::getSub(target, base);

      if (sizeTy != int32Ty) {
        offset = toolchain::ConstantExpr::getTrunc(offset, int32Ty);
      }
      offsets.push_back(offset);
    }
    auto constant = toolchain::ConstantArray::get(intArrayTy, offsets);
    global->setInitializer(constant);
    appendToUsed(M, global);

    toolchain::GlobalVariable::SanitizerMetadata Meta;
    Meta.IsDynInit = false;
    Meta.NoAddress = true;
    global->setSanitizerMetadata(Meta);

    changed = true;
  };

  addSection("__TEXT,__language_as_entry, coalesced, no_dead_strip",
             "__language_async_entry_functlets",
             asyncEntries);
  addSection("__TEXT,__language_as_ret, coalesced, no_dead_strip",
             "__language_async_ret_functlets",
             asyncReturns);

  if (!changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
