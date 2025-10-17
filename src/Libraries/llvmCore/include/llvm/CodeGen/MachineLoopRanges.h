/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

//===- MachineLoopRanges.h - Ranges of machine loops -----------*- c++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface to the MachineLoopRanges analysis.
//
// Provide on-demand information about the ranges of machine instructions
// covered by a loop.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINELOOPRANGES_H
#define LLVM_CODEGEN_MACHINELOOPRANGES_H

#include "llvm/ADT/IntervalMap.h"
#include "llvm/CodeGen/SlotIndexes.h"

namespace llvm {

class MachineLoop;
class MachineLoopInfo;
class raw_ostream;

/// MachineLoopRange - Range information for a single loop.
class MachineLoopRange {
  friend class MachineLoopRanges;

public:
  typedef IntervalMap<SlotIndex, unsigned, 4> Map;
  typedef Map::Allocator Allocator;

private:
  /// The mapped loop.
  const MachineLoop *const Loop;

  /// Map intervals to a bit mask.
  /// Bit 0 = inside loop block.
  Map Intervals;

  /// Loop area as measured by SlotIndex::distance.
  unsigned Area;

  /// Create a MachineLoopRange, only accessible to MachineLoopRanges.
  MachineLoopRange(const MachineLoop*, Allocator&, SlotIndexes&);

public:
  /// getLoop - Return the mapped machine loop.
  const MachineLoop *getLoop() const { return Loop; }

  /// overlaps - Return true if this loop overlaps the given range of machine
  /// inteructions.
  bool overlaps(SlotIndex Start, SlotIndex Stop);

  /// getNumber - Return the loop number. This is the same as the number of the
  /// header block.
  unsigned getNumber() const;

  /// getArea - Return the loop area. This number is approximately proportional
  /// to the number of instructions in the loop.
  unsigned getArea() const { return Area; }

  /// getMap - Allow public read-only access for IntervalMapOverlaps.
  const Map &getMap() { return Intervals; }

  /// print - Print loop ranges on OS.
  void print(raw_ostream&) const;

  /// byNumber - Comparator for array_pod_sort that sorts a list of
  /// MachineLoopRange pointers by number.
  static int byNumber(const void*, const void*);

  /// byAreaDesc - Comparator for array_pod_sort that sorts a list of
  /// MachineLoopRange pointers by descending area, then by number.
  static int byAreaDesc(const void*, const void*);
};

raw_ostream &operator<<(raw_ostream&, const MachineLoopRange&);

/// MachineLoopRanges - Analysis pass that provides on-demand per-loop range
/// information.
class MachineLoopRanges : public MachineFunctionPass {
  typedef DenseMap<const MachineLoop*, MachineLoopRange*> CacheMap;
  typedef MachineLoopRange::Allocator MapAllocator;

  MapAllocator Allocator;
  SlotIndexes *Indexes;
  CacheMap Cache;

public:
  static char ID; // Pass identification, replacement for typeid

  MachineLoopRanges() : MachineFunctionPass(ID), Indexes(0) {}
  ~MachineLoopRanges() { releaseMemory(); }

  /// getLoopRange - Return the range of loop.
  MachineLoopRange *getLoopRange(const MachineLoop *Loop);

private:
  virtual bool runOnMachineFunction(MachineFunction&);
  virtual void releaseMemory();
  virtual void getAnalysisUsage(AnalysisUsage&) const;
};


} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINELOOPRANGES_H
