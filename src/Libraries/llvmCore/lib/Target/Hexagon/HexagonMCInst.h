/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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

//===- HexagonMCInst.h - Hexagon sub-class of MCInst ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class extends MCInst to allow some VLIW annotation.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONMCINST_H
#define HEXAGONMCINST_H

#include "llvm/MC/MCInst.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {
  class HexagonMCInst: public MCInst {
    // Packet start and end markers
    unsigned startPacket: 1, endPacket: 1;
    const MachineInstr *MachineI;
  public:
    explicit HexagonMCInst(): MCInst(),
                              startPacket(0), endPacket(0) {}

    const MachineInstr* getMI() const { return MachineI; }

    void setMI(const MachineInstr *MI) { MachineI = MI; }

    bool isStartPacket() const { return (startPacket); }
    bool isEndPacket() const { return (endPacket); }

    void setStartPacket(bool yes) { startPacket = yes; }
    void setEndPacket(bool yes) { endPacket = yes; }
  };
}

#endif
