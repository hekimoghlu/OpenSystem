/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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

//===-- ARMHazardRecognizer.h - ARM Hazard Recognizers ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling ARM functions.
//
//===----------------------------------------------------------------------===//

#ifndef ARMHAZARDRECOGNIZER_H
#define ARMHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScoreboardHazardRecognizer.h"

namespace llvm {

class ARMBaseInstrInfo;
class ARMBaseRegisterInfo;
class ARMSubtarget;
class MachineInstr;

/// ARMHazardRecognizer handles special constraints that are not expressed in
/// the scheduling itinerary. This is only used during postRA scheduling. The
/// ARM preRA scheduler uses an unspecialized instance of the
/// ScoreboardHazardRecognizer.
class ARMHazardRecognizer : public ScoreboardHazardRecognizer {
  const ARMBaseInstrInfo &TII;
  const ARMBaseRegisterInfo &TRI;
  const ARMSubtarget &STI;

  MachineInstr *LastMI;
  unsigned FpMLxStalls;

public:
  ARMHazardRecognizer(const InstrItineraryData *ItinData,
                      const ARMBaseInstrInfo &tii,
                      const ARMBaseRegisterInfo &tri,
                      const ARMSubtarget &sti,
                      const ScheduleDAG *DAG) :
    ScoreboardHazardRecognizer(ItinData, DAG, "post-RA-sched"), TII(tii),
    TRI(tri), STI(sti), LastMI(0) {}

  virtual HazardType getHazardType(SUnit *SU, int Stalls);
  virtual void Reset();
  virtual void EmitInstruction(SUnit *SU);
  virtual void AdvanceCycle();
  virtual void RecedeCycle();
};

} // end namespace llvm

#endif // ARMHAZARDRECOGNIZER_H
