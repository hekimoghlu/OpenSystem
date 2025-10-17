/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

//===- NVPTXSubtarget.cpp - NVPTX Subtarget Information -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVPTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "NVPTXSubtarget.h"
#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "NVPTXGenSubtargetInfo.inc"

using namespace llvm;

// Select Driver Interface
#include "llvm/Support/CommandLine.h"
namespace {
cl::opt<NVPTX::DrvInterface>
DriverInterface(cl::desc("Choose driver interface:"),
                cl::values(
                    clEnumValN(NVPTX::NVCL, "drvnvcl", "Nvidia OpenCL driver"),
                    clEnumValN(NVPTX::CUDA, "drvcuda", "Nvidia CUDA driver"),
                    clEnumValN(NVPTX::TEST, "drvtest", "Plain Test"),
                    clEnumValEnd),
                    cl::init(NVPTX::NVCL));
}

NVPTXSubtarget::NVPTXSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS, bool is64Bit)
:NVPTXGenSubtargetInfo(TT, "", FS), // Don't pass CPU to subtarget,
 // because we don't register all
 // nvptx targets.
 Is64Bit(is64Bit) {

  drvInterface = DriverInterface;

  // Provide the default CPU if none
  std::string defCPU = "sm_10";

  // Get the TargetName from the FS if available
  if (FS.empty() && CPU.empty())
    TargetName = defCPU;
  else if (!CPU.empty())
    TargetName = CPU;
  else
    llvm_unreachable("we are not using FeatureStr");

  // Set up the SmVersion
  SmVersion = atoi(TargetName.c_str()+3);
}
