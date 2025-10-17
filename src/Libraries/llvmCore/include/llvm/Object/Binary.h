/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

//===- Binary.h - A generic binary file -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Binary class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_BINARY_H
#define LLVM_OBJECT_BINARY_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Object/Error.h"

namespace llvm {

class MemoryBuffer;
class StringRef;

namespace object {

class Binary {
private:
  Binary() LLVM_DELETED_FUNCTION;
  Binary(const Binary &other) LLVM_DELETED_FUNCTION;

  unsigned int TypeID;

protected:
  MemoryBuffer *Data;

  Binary(unsigned int Type, MemoryBuffer *Source);

  enum {
    ID_Archive,
    // Object and children.
    ID_StartObjects,
    ID_COFF,
    ID_ELF32L, // ELF 32-bit, little endian
    ID_ELF32B, // ELF 32-bit, big endian
    ID_ELF64L, // ELF 64-bit, little endian
    ID_ELF64B, // ELF 64-bit, big endian
    ID_MachO,
    ID_EndObjects
  };

  static inline unsigned int getELFType(bool isLittleEndian, bool is64Bits) {
    if (isLittleEndian)
      return is64Bits ? ID_ELF64L : ID_ELF32L;
    else
      return is64Bits ? ID_ELF64B : ID_ELF32B;
  }

public:
  virtual ~Binary();

  StringRef getData() const;
  StringRef getFileName() const;

  // Cast methods.
  unsigned int getType() const { return TypeID; }
  static inline bool classof(const Binary *v) { return true; }

  // Convenience methods
  bool isObject() const {
    return TypeID > ID_StartObjects && TypeID < ID_EndObjects;
  }

  bool isArchive() const {
    return TypeID == ID_Archive;
  }

  bool isELF() const {
    return TypeID >= ID_ELF32L && TypeID <= ID_ELF64B;
  }

  bool isMachO() const {
    return TypeID == ID_MachO;
  }

  bool isCOFF() const {
    return TypeID == ID_COFF;
  }
};

/// @brief Create a Binary from Source, autodetecting the file type.
///
/// @param Source The data to create the Binary from. Ownership is transferred
///        to Result if successful. If an error is returned, Source is destroyed
///        by createBinary before returning.
/// @param Result A pointer to the resulting Binary if no error occured.
error_code createBinary(MemoryBuffer *Source, OwningPtr<Binary> &Result);

error_code createBinary(StringRef Path, OwningPtr<Binary> &Result);

}
}

#endif
