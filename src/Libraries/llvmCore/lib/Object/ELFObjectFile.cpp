/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

//===- ELFObjectFile.cpp - ELF object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the ELFObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"

namespace llvm {

using namespace object;

// Creates an in-memory object-file by default: createELFObjectFile(Buffer)
ObjectFile *ObjectFile::createELFObjectFile(MemoryBuffer *Object) {
  std::pair<unsigned char, unsigned char> Ident = getElfArchType(Object);
  error_code ec;

  if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2LSB)
    return new ELFObjectFile<support::little, false>(Object, ec);
  else if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2MSB)
    return new ELFObjectFile<support::big, false>(Object, ec);
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2MSB)
    return new ELFObjectFile<support::big, true>(Object, ec);
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2LSB) {
    ELFObjectFile<support::little, true> *result =
          new ELFObjectFile<support::little, true>(Object, ec);
    return result;
  }

  report_fatal_error("Buffer is not an ELF object file!");
}

} // end namespace llvm
