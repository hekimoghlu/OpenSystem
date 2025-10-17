/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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

//===-- HexagonSelectionDAGInfo.cpp - Hexagon SelectionDAG Info -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HexagonSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hexagon-selectiondag-info"
#include "HexagonTargetMachine.h"
using namespace llvm;

bool llvm::flag_aligned_memcpy;

HexagonSelectionDAGInfo::HexagonSelectionDAGInfo(const HexagonTargetMachine
                                                 &TM)
  : TargetSelectionDAGInfo(TM) {
}

HexagonSelectionDAGInfo::~HexagonSelectionDAGInfo() {
}

SDValue
HexagonSelectionDAGInfo::
EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl, SDValue Chain,
                        SDValue Dst, SDValue Src, SDValue Size, unsigned Align,
                        bool isVolatile, bool AlwaysInline,
                        MachinePointerInfo DstPtrInfo,
                        MachinePointerInfo SrcPtrInfo) const {
  flag_aligned_memcpy = false;
  if ((Align & 0x3) == 0) {
    ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
    if (ConstantSize) {
      uint64_t SizeVal = ConstantSize->getZExtValue();
      if ((SizeVal > 32) && ((SizeVal % 8) == 0))
        flag_aligned_memcpy = true;
    }
  }

  return SDValue();
}
