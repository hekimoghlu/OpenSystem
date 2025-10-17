/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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

//===-- MipsTargetInfo.cpp - Mips Target Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "llvm/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheMipsTarget, llvm::TheMipselTarget;
Target llvm::TheMips64Target, llvm::TheMips64elTarget;

extern "C" void LLVMInitializeMipsTargetInfo() {
  RegisterTarget<Triple::mips,
        /*HasJIT=*/true> X(TheMipsTarget, "mips", "Mips");

  RegisterTarget<Triple::mipsel,
        /*HasJIT=*/true> Y(TheMipselTarget, "mipsel", "Mipsel");

  RegisterTarget<Triple::mips64,
        /*HasJIT=*/false> A(TheMips64Target, "mips64", "Mips64 [experimental]");

  RegisterTarget<Triple::mips64el,
        /*HasJIT=*/false> B(TheMips64elTarget,
                            "mips64el", "Mips64el [experimental]");
}
