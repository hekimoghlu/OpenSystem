/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

//===-- RuntimeDyldMachO.h - Run-time dynamic linker for MC-JIT ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachO support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_MACHO_H
#define LLVM_RUNTIME_DYLD_MACHO_H

#include "llvm/ADT/IndexedMap.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/Format.h"
#include "RuntimeDyldImpl.h"

using namespace llvm;
using namespace llvm::object;


namespace llvm {
class RuntimeDyldMachO : public RuntimeDyldImpl {
protected:
  bool resolveI386Relocation(uint8_t *LocalAddress,
                             uint64_t FinalAddress,
                             uint64_t Value,
                             bool isPCRel,
                             unsigned Type,
                             unsigned Size,
                             int64_t Addend);
  bool resolveX86_64Relocation(uint8_t *LocalAddress,
                               uint64_t FinalAddress,
                               uint64_t Value,
                               bool isPCRel,
                               unsigned Type,
                               unsigned Size,
                               int64_t Addend);
  bool resolveARMRelocation(uint8_t *LocalAddress,
                            uint64_t FinalAddress,
                            uint64_t Value,
                            bool isPCRel,
                            unsigned Type,
                            unsigned Size,
                            int64_t Addend);

  virtual void processRelocationRef(const ObjRelocationInfo &Rel,
                                    ObjectImage &Obj,
                                    ObjSectionToIDMap &ObjSectionToID,
                                    const SymbolTableMap &Symbols,
                                    StubMap &Stubs);

public:
  virtual void resolveRelocation(uint8_t *LocalAddress,
                                 uint64_t FinalAddress,
                                 uint64_t Value,
                                 uint32_t Type,
                                 int64_t Addend);

  RuntimeDyldMachO(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const;
};

} // end namespace llvm

#endif
