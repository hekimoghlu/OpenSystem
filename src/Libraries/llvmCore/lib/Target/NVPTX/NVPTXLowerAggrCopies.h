/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

//===-- llvm/lib/Target/NVPTX/NVPTXLowerAggrCopies.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVIDIA specific lowering of
// aggregate copies
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_LOWER_AGGR_COPIES_H
#define NVPTX_LOWER_AGGR_COPIES_H

#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

// actual analysis class, which is a functionpass
struct NVPTXLowerAggrCopies : public FunctionPass {
  static char ID;

  NVPTXLowerAggrCopies() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<TargetData>();
    AU.addPreserved<MachineFunctionAnalysis>();
  }

  virtual bool runOnFunction(Function &F);

  static const unsigned MaxAggrCopySize = 128;

  virtual const char *getPassName() const {
    return "Lower aggregate copies/intrinsics into loops";
  }
};

extern FunctionPass *createLowerAggrCopies();
}

#endif
