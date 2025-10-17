/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

//===-- RuntimeDyldELF.h - Run-time dynamic linker for MC-JIT ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ELF support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_ELF_H
#define LLVM_RUNTIME_DYLD_ELF_H

#include "RuntimeDyldImpl.h"

using namespace llvm;


namespace llvm {
class RuntimeDyldELF : public RuntimeDyldImpl {
protected:
  ObjectImage *LoadedObject;

  void resolveX86_64Relocation(uint8_t *LocalAddress,
                               uint64_t FinalAddress,
                               uint64_t Value,
                               uint32_t Type,
                               int64_t Addend);

  void resolveX86Relocation(uint8_t *LocalAddress,
                            uint32_t FinalAddress,
                            uint32_t Value,
                            uint32_t Type,
                            int32_t Addend);

  void resolveARMRelocation(uint8_t *LocalAddress,
                            uint32_t FinalAddress,
                            uint32_t Value,
                            uint32_t Type,
                            int32_t Addend);

  void resolveMIPSRelocation(uint8_t *LocalAddress,
                             uint32_t FinalAddress,
                             uint32_t Value,
                             uint32_t Type,
                             int32_t Addend);

  virtual void resolveRelocation(uint8_t *LocalAddress,
                                 uint64_t FinalAddress,
                                 uint64_t Value,
                                 uint32_t Type,
                                 int64_t Addend);

  virtual void processRelocationRef(const ObjRelocationInfo &Rel,
                                    ObjectImage &Obj,
                                    ObjSectionToIDMap &ObjSectionToID,
                                    const SymbolTableMap &Symbols,
                                    StubMap &Stubs);

  virtual ObjectImage *createObjectImage(const MemoryBuffer *InputBuffer);
  virtual void handleObjectLoaded(ObjectImage *Obj);

public:
  RuntimeDyldELF(RTDyldMemoryManager *mm)
      : RuntimeDyldImpl(mm), LoadedObject(0) {}

  virtual ~RuntimeDyldELF();

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const;
};

} // end namespace llvm

#endif
