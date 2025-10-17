/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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

//===-- CodeGen/MachineInstr.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the machine function pass registry for register allocators
// and instruction schedulers.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePassRegistry.h"

using namespace llvm;

void MachinePassRegistryListener::anchor() { }

/// setDefault - Set the default constructor by name.
void MachinePassRegistry::setDefault(StringRef Name) {
  MachinePassCtor Ctor = 0;
  for(MachinePassRegistryNode *R = getList(); R; R = R->getNext()) {
    if (R->getName() == Name) {
      Ctor = R->getCtor();
      break;
    }
  }
  assert(Ctor && "Unregistered pass name");
  setDefault(Ctor);
}

/// Add - Adds a function pass to the registration list.
///
void MachinePassRegistry::Add(MachinePassRegistryNode *Node) {
  Node->setNext(List);
  List = Node;
  if (Listener) Listener->NotifyAdd(Node->getName(),
                                    Node->getCtor(),
                                    Node->getDescription());
}


/// Remove - Removes a function pass from the registration list.
///
void MachinePassRegistry::Remove(MachinePassRegistryNode *Node) {
  for (MachinePassRegistryNode **I = &List; *I; I = (*I)->getNextAddress()) {
    if (*I == Node) {
      if (Listener) Listener->NotifyRemove(Node->getName());
      *I = (*I)->getNext();
      break;
    }
  }
}
