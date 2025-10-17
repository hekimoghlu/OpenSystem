/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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

//===-- XCore.h - Top-level interface for XCore representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// XCore back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_XCORE_H
#define TARGET_XCORE_H

#include "MCTargetDesc/XCoreMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class FunctionPass;
  class TargetMachine;
  class XCoreTargetMachine;
  class formatted_raw_ostream;

  FunctionPass *createXCoreISelDag(XCoreTargetMachine &TM,
                                   CodeGenOpt::Level OptLevel);

} // end namespace llvm;

#endif
