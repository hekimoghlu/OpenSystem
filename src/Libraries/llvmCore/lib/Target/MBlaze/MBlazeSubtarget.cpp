/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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

//===-- MBlazeSubtarget.cpp - MBlaze Subtarget Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MBlaze specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "MBlazeSubtarget.h"
#include "MBlaze.h"
#include "MBlazeRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MBlazeGenSubtargetInfo.inc"

using namespace llvm;

MBlazeSubtarget::MBlazeSubtarget(const std::string &TT,
                                 const std::string &CPU,
                                 const std::string &FS):
  MBlazeGenSubtargetInfo(TT, CPU, FS),
  HasBarrel(false), HasDiv(false), HasMul(false), HasPatCmp(false),
  HasFPU(false), HasMul64(false), HasSqrt(false)
{
  // Parse features string.
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "mblaze";
  ParseSubtargetFeatures(CPUName, FS);

  // Only use instruction scheduling if the selected CPU has an instruction
  // itinerary (the default CPU is the only one that doesn't).
  HasItin = CPUName != "mblaze";
  DEBUG(dbgs() << "CPU " << CPUName << "(" << HasItin << ")\n");

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);
}

bool MBlazeSubtarget::
enablePostRAScheduler(CodeGenOpt::Level OptLevel,
                      TargetSubtargetInfo::AntiDepBreakMode& Mode,
                      RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtargetInfo::ANTIDEP_CRITICAL;
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(&MBlaze::GPRRegClass);
  return HasItin && OptLevel >= CodeGenOpt::Default;
}
