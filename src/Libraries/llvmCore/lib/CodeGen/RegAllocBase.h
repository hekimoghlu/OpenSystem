/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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

//===-- RegAllocBase.h - basic regalloc interface and driver --*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegAllocBase class, which is the skeleton of a basic
// register allocation algorithm and interface for extending it. It provides the
// building blocks on which to construct other experimental allocators and test
// the validity of two principles:
//
// - If virtual and physical register liveness is modeled using intervals, then
// on-the-fly interference checking is cheap. Furthermore, interferences can be
// lazily cached and reused.
//
// - Register allocation complexity, and generated code performance is
// determined by the effectiveness of live range splitting rather than optimal
// coloring.
//
// Following the first principle, interfering checking revolves around the
// LiveIntervalUnion data structure.
//
// To fulfill the second principle, the basic allocator provides a driver for
// incremental splitting. It essentially punts on the problem of register
// coloring, instead driving the assignment of virtual to physical registers by
// the cost of splitting. The basic allocator allows for heuristic reassignment
// of registers, if a more sophisticated allocator chooses to do that.
//
// This framework provides a way to engineer the compile time vs. code
// quality trade-off without relying on a particular theoretical solver.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCBASE
#define LLVM_CODEGEN_REGALLOCBASE

#include "LiveIntervalUnion.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/ADT/OwningPtr.h"

namespace llvm {

template<typename T> class SmallVectorImpl;
class TargetRegisterInfo;
class VirtRegMap;
class LiveIntervals;
class LiveRegMatrix;
class Spiller;

/// RegAllocBase provides the register allocation driver and interface that can
/// be extended to add interesting heuristics.
///
/// Register allocators must override the selectOrSplit() method to implement
/// live range splitting. They must also override enqueue/dequeue to provide an
/// assignment order.
class RegAllocBase {
protected:
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  VirtRegMap *VRM;
  LiveIntervals *LIS;
  LiveRegMatrix *Matrix;
  RegisterClassInfo RegClassInfo;

  RegAllocBase(): TRI(0), MRI(0), VRM(0), LIS(0), Matrix(0) {}

  virtual ~RegAllocBase() {}

  // A RegAlloc pass should call this before allocatePhysRegs.
  void init(VirtRegMap &vrm, LiveIntervals &lis, LiveRegMatrix &mat);

  // The top-level driver. The output is a VirtRegMap that us updated with
  // physical register assignments.
  void allocatePhysRegs();

  // Get a temporary reference to a Spiller instance.
  virtual Spiller &spiller() = 0;

  /// enqueue - Add VirtReg to the priority queue of unassigned registers.
  virtual void enqueue(LiveInterval *LI) = 0;

  /// dequeue - Return the next unassigned register, or NULL.
  virtual LiveInterval *dequeue() = 0;

  // A RegAlloc pass should override this to provide the allocation heuristics.
  // Each call must guarantee forward progess by returning an available PhysReg
  // or new set of split live virtual registers. It is up to the splitter to
  // converge quickly toward fully spilled live ranges.
  virtual unsigned selectOrSplit(LiveInterval &VirtReg,
                                 SmallVectorImpl<LiveInterval*> &splitLVRs) = 0;

  // Use this group name for NamedRegionTimer.
  static const char *TimerGroupName;

public:
  /// VerifyEnabled - True when -verify-regalloc is given.
  static bool VerifyEnabled;

private:
  void seedLiveRegs();
};

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_REGALLOCBASE)
