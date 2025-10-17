/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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

//===-- llvm/CodeGen/AllocationOrder.cpp - Allocation Order ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an allocation order for virtual registers.
//
// The preferred allocation order for a virtual register depends on allocation
// hints and target hooks. The AllocationOrder class encapsulates all of that.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

using namespace llvm;

// Compare VirtRegMap::getRegAllocPref().
AllocationOrder::AllocationOrder(unsigned VirtReg,
                                 const VirtRegMap &VRM,
                                 const RegisterClassInfo &RegClassInfo)
  : Begin(0), End(0), Pos(0), RCI(RegClassInfo), OwnedBegin(false) {
  const TargetRegisterClass *RC = VRM.getRegInfo().getRegClass(VirtReg);
  std::pair<unsigned, unsigned> HintPair =
    VRM.getRegInfo().getRegAllocationHint(VirtReg);
  const MachineRegisterInfo &MRI = VRM.getRegInfo();

  // HintPair.second is a register, phys or virt.
  Hint = HintPair.second;

  // Translate to physreg, or 0 if not assigned yet.
  if (TargetRegisterInfo::isVirtualRegister(Hint))
    Hint = VRM.getPhys(Hint);

  // The first hint pair component indicates a target-specific hint.
  if (HintPair.first) {
    const TargetRegisterInfo &TRI = VRM.getTargetRegInfo();
    // The remaining allocation order may depend on the hint.
    ArrayRef<uint16_t> Order =
      TRI.getRawAllocationOrder(RC, HintPair.first, Hint,
                                VRM.getMachineFunction());
    if (Order.empty())
      return;

    // Copy the allocation order with reserved registers removed.
    OwnedBegin = true;
    unsigned *P = new unsigned[Order.size()];
    Begin = P;
    for (unsigned i = 0; i != Order.size(); ++i)
      if (!MRI.isReserved(Order[i]))
        *P++ = Order[i];
    End = P;

    // Target-dependent hints require resolution.
    Hint = TRI.ResolveRegAllocHint(HintPair.first, Hint,
                                   VRM.getMachineFunction());
  } else {
    // If there is no hint or just a normal hint, use the cached allocation
    // order from RegisterClassInfo.
    ArrayRef<unsigned> O = RCI.getOrder(RC);
    Begin = O.begin();
    End = O.end();
  }

  // The hint must be a valid physreg for allocation.
  if (Hint && (!TargetRegisterInfo::isPhysicalRegister(Hint) ||
               !RC->contains(Hint) || MRI.isReserved(Hint)))
    Hint = 0;
}

AllocationOrder::~AllocationOrder() {
  if (OwnedBegin)
    delete [] Begin;
}
