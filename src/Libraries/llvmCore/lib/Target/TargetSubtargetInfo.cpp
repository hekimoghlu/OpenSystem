/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

//===-- TargetSubtargetInfo.cpp - General Target Information ---------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Subtarget.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetSubtargetInfo Class
//
TargetSubtargetInfo::TargetSubtargetInfo() {}

TargetSubtargetInfo::~TargetSubtargetInfo() {}

bool TargetSubtargetInfo::enablePostRAScheduler(
          CodeGenOpt::Level OptLevel,
          AntiDepBreakMode& Mode,
          RegClassVector& CriticalPathRCs) const {
  Mode = ANTIDEP_NONE;
  CriticalPathRCs.clear();
  return false;
}

