/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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

//===-- X86SelectionDAGInfo.h - X86 SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef X86SELECTIONDAGINFO_H
#define X86SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class X86TargetLowering;
class X86TargetMachine;
class X86Subtarget;

class X86SelectionDAGInfo : public TargetSelectionDAGInfo {
  /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const X86Subtarget *Subtarget;

  const X86TargetLowering &TLI;

public:
  explicit X86SelectionDAGInfo(const X86TargetMachine &TM);
  ~X86SelectionDAGInfo();

  virtual
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align,
                                  bool isVolatile,
                                  MachinePointerInfo DstPtrInfo) const;

  virtual
  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const;
};

}

#endif
