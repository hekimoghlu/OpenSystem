/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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

//===-- llvm/lib/Target/NVPTX/NVPTXSplitBBatBar.h ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVIDIA specific declarations
// for splitting basic blocks at barrier instructions.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_SPLIT_BB_AT_BAR_H
#define NVPTX_SPLIT_BB_AT_BAR_H

#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"

namespace llvm {

// actual analysis class, which is a functionpass
struct NVPTXSplitBBatBar : public FunctionPass {
  static char ID;

  NVPTXSplitBBatBar() : FunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addPreserved<MachineFunctionAnalysis>();
  }
  virtual bool runOnFunction(Function &F);

  virtual const char *getPassName() const {
    return "Split basic blocks at barrier";
  }
};

extern FunctionPass *createSplitBBatBarPass();
}

#endif //NVPTX_SPLIT_BB_AT_BAR_H
