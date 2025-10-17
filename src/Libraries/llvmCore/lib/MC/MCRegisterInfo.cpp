/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

//=== MC/MCRegisterInfo.cpp - Target Register Description -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements MCRegisterInfo functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;

unsigned MCRegisterInfo::getMatchingSuperReg(unsigned Reg, unsigned SubIdx,
                                             const MCRegisterClass *RC) const {
  for (MCSuperRegIterator Supers(Reg, this); Supers.isValid(); ++Supers)
    if (RC->contains(*Supers) && Reg == getSubReg(*Supers, SubIdx))
      return *Supers;
  return 0;
}

unsigned MCRegisterInfo::getSubReg(unsigned Reg, unsigned Idx) const {
  assert(Idx && Idx < getNumSubRegIndices() &&
         "This is not a subregister index");
  // Get a pointer to the corresponding SubRegIndices list. This list has the
  // name of each sub-register in the same order as MCSubRegIterator.
  const uint16_t *SRI = SubRegIndices + get(Reg).SubRegIndices;
  for (MCSubRegIterator Subs(Reg, this); Subs.isValid(); ++Subs, ++SRI)
    if (*SRI == Idx)
      return *Subs;
  return 0;
}

unsigned MCRegisterInfo::getSubRegIndex(unsigned Reg, unsigned SubReg) const {
  assert(SubReg && SubReg < getNumRegs() && "This is not a register");
  // Get a pointer to the corresponding SubRegIndices list. This list has the
  // name of each sub-register in the same order as MCSubRegIterator.
  const uint16_t *SRI = SubRegIndices + get(Reg).SubRegIndices;
  for (MCSubRegIterator Subs(Reg, this); Subs.isValid(); ++Subs, ++SRI)
    if (*Subs == SubReg)
      return *SRI;
  return 0;
}

int MCRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  const DwarfLLVMRegPair *M = isEH ? EHL2DwarfRegs : L2DwarfRegs;
  unsigned Size = isEH ? EHL2DwarfRegsSize : L2DwarfRegsSize;

  DwarfLLVMRegPair Key = { RegNum, 0 };
  const DwarfLLVMRegPair *I = std::lower_bound(M, M+Size, Key);
  if (I == M+Size || I->FromReg != RegNum)
    return -1;
  return I->ToReg;
}

int MCRegisterInfo::getLLVMRegNum(unsigned RegNum, bool isEH) const {
  const DwarfLLVMRegPair *M = isEH ? EHDwarf2LRegs : Dwarf2LRegs;
  unsigned Size = isEH ? EHDwarf2LRegsSize : Dwarf2LRegsSize;

  DwarfLLVMRegPair Key = { RegNum, 0 };
  const DwarfLLVMRegPair *I = std::lower_bound(M, M+Size, Key);
  assert(I != M+Size && I->FromReg == RegNum && "Invalid RegNum");
  return I->ToReg;
}

int MCRegisterInfo::getSEHRegNum(unsigned RegNum) const {
  const DenseMap<unsigned, int>::const_iterator I = L2SEHRegs.find(RegNum);
  if (I == L2SEHRegs.end()) return (int)RegNum;
  return I->second;
}
