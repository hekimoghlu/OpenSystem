/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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

//===-- llvm/CodeGen/RegAllocRegistry.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for register allocator function
// pass registry (RegisterRegAlloc).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGENREGALLOCREGISTRY_H
#define LLVM_CODEGENREGALLOCREGISTRY_H

#include "llvm/CodeGen/MachinePassRegistry.h"

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//===----------------------------------------------------------------------===//
class RegisterRegAlloc : public MachinePassRegistryNode {

public:

  typedef FunctionPass *(*FunctionPassCtor)();

  static MachinePassRegistry Registry;

  RegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
  : MachinePassRegistryNode(N, D, (MachinePassCtor)C)
  { 
     Registry.Add(this); 
  }
  ~RegisterRegAlloc() { Registry.Remove(this); }
  

  // Accessors.
  //
  RegisterRegAlloc *getNext() const {
    return (RegisterRegAlloc *)MachinePassRegistryNode::getNext();
  }
  static RegisterRegAlloc *getList() {
    return (RegisterRegAlloc *)Registry.getList();
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

} // end namespace llvm


#endif
