/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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

//===-- llvm/CodeGen/SchedulerRegistry.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for instruction scheduler function
// pass registry (RegisterScheduler).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGENSCHEDULERREGISTRY_H
#define LLVM_CODEGENSCHEDULERREGISTRY_H

#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===----------------------------------------------------------------------===//

class SelectionDAGISel;
class ScheduleDAGSDNodes;
class SelectionDAG;
class MachineBasicBlock;

class RegisterScheduler : public MachinePassRegistryNode {
public:
  typedef ScheduleDAGSDNodes *(*FunctionPassCtor)(SelectionDAGISel*,
                                                  CodeGenOpt::Level);

  static MachinePassRegistry Registry;

  RegisterScheduler(const char *N, const char *D, FunctionPassCtor C)
  : MachinePassRegistryNode(N, D, (MachinePassCtor)C)
  { Registry.Add(this); }
  ~RegisterScheduler() { Registry.Remove(this); }


  // Accessors.
  //
  RegisterScheduler *getNext() const {
    return (RegisterScheduler *)MachinePassRegistryNode::getNext();
  }
  static RegisterScheduler *getList() {
    return (RegisterScheduler *)Registry.getList();
  }
  static FunctionPassCtor getDefault() {
    return (FunctionPassCtor)Registry.getDefault();
  }
  static void setDefault(FunctionPassCtor C) {
    Registry.setDefault((MachinePassCtor)C);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
};

/// createBURRListDAGScheduler - This creates a bottom up register usage
/// reduction list scheduler.
ScheduleDAGSDNodes *createBURRListDAGScheduler(SelectionDAGISel *IS,
                                               CodeGenOpt::Level OptLevel);

/// createBURRListDAGScheduler - This creates a bottom up list scheduler that
/// schedules nodes in source code order when possible.
ScheduleDAGSDNodes *createSourceListDAGScheduler(SelectionDAGISel *IS,
                                                 CodeGenOpt::Level OptLevel);

/// createHybridListDAGScheduler - This creates a bottom up register pressure
/// aware list scheduler that make use of latency information to avoid stalls
/// for long latency instructions in low register pressure mode. In high
/// register pressure mode it schedules to reduce register pressure.
ScheduleDAGSDNodes *createHybridListDAGScheduler(SelectionDAGISel *IS,
                                                 CodeGenOpt::Level);

/// createILPListDAGScheduler - This creates a bottom up register pressure
/// aware list scheduler that tries to increase instruction level parallelism
/// in low register pressure mode. In high register pressure mode it schedules
/// to reduce register pressure.
ScheduleDAGSDNodes *createILPListDAGScheduler(SelectionDAGISel *IS,
                                              CodeGenOpt::Level);

/// createFastDAGScheduler - This creates a "fast" scheduler.
///
ScheduleDAGSDNodes *createFastDAGScheduler(SelectionDAGISel *IS,
                                           CodeGenOpt::Level OptLevel);

/// createVLIWDAGScheduler - Scheduler for VLIW targets. This creates top down
/// DFA driven list scheduler with clustering heuristic to control
/// register pressure.
ScheduleDAGSDNodes *createVLIWDAGScheduler(SelectionDAGISel *IS,
                                           CodeGenOpt::Level OptLevel);
/// createDefaultScheduler - This creates an instruction scheduler appropriate
/// for the target.
ScheduleDAGSDNodes *createDefaultScheduler(SelectionDAGISel *IS,
                                           CodeGenOpt::Level OptLevel);

} // end namespace llvm

#endif
