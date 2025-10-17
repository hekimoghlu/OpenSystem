/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

//==-- llvm/MC/MCSubtargetInfo.h - Subtarget Information ---------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the subtarget options of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSUBTARGET_H
#define LLVM_MC_MCSUBTARGET_H

#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/MCInstrItineraries.h"
#include <string>

namespace llvm {

class StringRef;

//===----------------------------------------------------------------------===//
///
/// MCSubtargetInfo - Generic base class for all target subtargets.
///
class MCSubtargetInfo {
  std::string TargetTriple;            // Target triple
  const SubtargetFeatureKV *ProcFeatures;  // Processor feature list
  const SubtargetFeatureKV *ProcDesc;  // Processor descriptions

  // Scheduler machine model
  const SubtargetInfoKV *ProcSchedModels;
  const MCWriteProcResEntry *WriteProcResTable;
  const MCWriteLatencyEntry *WriteLatencyTable;
  const MCReadAdvanceEntry *ReadAdvanceTable;
  const MCSchedModel *CPUSchedModel;

  const InstrStage *Stages;            // Instruction itinerary stages
  const unsigned *OperandCycles;       // Itinerary operand cycles
  const unsigned *ForwardingPaths;     // Forwarding paths
  unsigned NumFeatures;                // Number of processor features
  unsigned NumProcs;                   // Number of processors
  uint64_t FeatureBits;                // Feature bits for current CPU + FS

public:
  void InitMCSubtargetInfo(StringRef TT, StringRef CPU, StringRef FS,
                           const SubtargetFeatureKV *PF,
                           const SubtargetFeatureKV *PD,
                           const SubtargetInfoKV *ProcSched,
                           const MCWriteProcResEntry *WPR,
                           const MCWriteLatencyEntry *WL,
                           const MCReadAdvanceEntry *RA,
                           const InstrStage *IS,
                           const unsigned *OC, const unsigned *FP,
                           unsigned NF, unsigned NP);

  /// getTargetTriple - Return the target triple string.
  StringRef getTargetTriple() const {
    return TargetTriple;
  }

  /// getFeatureBits - Return the feature bits.
  ///
  uint64_t getFeatureBits() const {
    return FeatureBits;
  }

  /// InitMCProcessorInfo - Set or change the CPU (optionally supplemented with
  /// feature string). Recompute feature bits and scheduling model.
  void InitMCProcessorInfo(StringRef CPU, StringRef FS);

  /// ToggleFeature - Toggle a feature and returns the re-computed feature
  /// bits. This version does not change the implied bits.
  uint64_t ToggleFeature(uint64_t FB);

  /// ToggleFeature - Toggle a feature and returns the re-computed feature
  /// bits. This version will also change all implied bits.
  uint64_t ToggleFeature(StringRef FS);

  /// getSchedModelForCPU - Get the machine model of a CPU.
  ///
  const MCSchedModel *getSchedModelForCPU(StringRef CPU) const;

  /// getSchedModel - Get the machine model for this subtarget's CPU.
  ///
  const MCSchedModel *getSchedModel() const { return CPUSchedModel; }

  /// Return an iterator at the first process resource consumed by the given
  /// scheduling class.
  const MCWriteProcResEntry *getWriteProcResBegin(
    const MCSchedClassDesc *SC) const {
    return &WriteProcResTable[SC->WriteProcResIdx];
  }
  const MCWriteProcResEntry *getWriteProcResEnd(
    const MCSchedClassDesc *SC) const {
    return getWriteProcResBegin(SC) + SC->NumWriteProcResEntries;
  }

  const MCWriteLatencyEntry *getWriteLatencyEntry(const MCSchedClassDesc *SC,
                                                  unsigned DefIdx) const {
    assert(DefIdx < SC->NumWriteLatencyEntries &&
           "MachineModel does not specify a WriteResource for DefIdx");

    return &WriteLatencyTable[SC->WriteLatencyIdx + DefIdx];
  }

  int getReadAdvanceCycles(const MCSchedClassDesc *SC, unsigned UseIdx,
                           unsigned WriteResID) const {
    // TODO: The number of read advance entries in a class can be significant
    // (~50). Consider compressing the WriteID into a dense ID of those that are
    // used by ReadAdvance and representing them as a bitset.
    for (const MCReadAdvanceEntry *I = &ReadAdvanceTable[SC->ReadAdvanceIdx],
           *E = I + SC->NumReadAdvanceEntries; I != E; ++I) {
      if (I->UseIdx < UseIdx)
        continue;
      if (I->UseIdx > UseIdx)
        break;
      // Find the first WriteResIdx match, which has the highest cycle count.
      if (!I->WriteResourceID || I->WriteResourceID == WriteResID) {
        return I->Cycles;
      }
    }
    return 0;
  }

  /// getInstrItineraryForCPU - Get scheduling itinerary of a CPU.
  ///
  InstrItineraryData getInstrItineraryForCPU(StringRef CPU) const;

  /// Initialize an InstrItineraryData instance.
  void initInstrItins(InstrItineraryData &InstrItins) const;
};

} // End llvm namespace

#endif
