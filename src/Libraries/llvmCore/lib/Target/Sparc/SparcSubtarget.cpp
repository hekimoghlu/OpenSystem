/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

//===-- SparcSubtarget.cpp - SPARC Subtarget Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPARC specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SparcSubtarget.h"
#include "Sparc.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SparcGenSubtargetInfo.inc"

using namespace llvm;

void SparcSubtarget::anchor() { }

SparcSubtarget::SparcSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS,  bool is64Bit) :
  SparcGenSubtargetInfo(TT, CPU, FS),
  IsV9(false),
  V8DeprecatedInsts(false),
  IsVIS(false),
  Is64Bit(is64Bit) {
  
  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty()) {
    if (is64Bit)
      CPUName = "v9";
    else
      CPUName = "v8";
  }
  IsV9 = CPUName == "v9";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
}
