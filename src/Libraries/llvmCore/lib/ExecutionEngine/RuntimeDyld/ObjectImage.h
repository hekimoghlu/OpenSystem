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

//===---- ObjectImage.h - Format independent executuable object image -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a file format independent ObjectImage class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H
#define LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H

#include "llvm/Object/ObjectFile.h"

namespace llvm {

class ObjectImage {
  ObjectImage() LLVM_DELETED_FUNCTION;
  ObjectImage(const ObjectImage &other) LLVM_DELETED_FUNCTION;
protected:
  object::ObjectFile *ObjFile;

public:
  ObjectImage(object::ObjectFile *Obj) { ObjFile = Obj; }
  virtual ~ObjectImage() {}

  virtual object::symbol_iterator begin_symbols() const
              { return ObjFile->begin_symbols(); }
  virtual object::symbol_iterator end_symbols() const
              { return ObjFile->end_symbols(); }

  virtual object::section_iterator begin_sections() const
              { return ObjFile->begin_sections(); }
  virtual object::section_iterator end_sections() const
              { return ObjFile->end_sections(); }

  virtual /* Triple::ArchType */ unsigned getArch() const
              { return ObjFile->getArch(); }

  // Subclasses can override these methods to update the image with loaded
  // addresses for sections and common symbols
  virtual void updateSectionAddress(const object::SectionRef &Sec,
                                    uint64_t Addr) {}
  virtual void updateSymbolAddress(const object::SymbolRef &Sym, uint64_t Addr)
              {}

  // Subclasses can override these methods to provide JIT debugging support
  virtual void registerWithDebugger() {}
  virtual void deregisterWithDebugger() {}
};

} // end namespace llvm

#endif // LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H

