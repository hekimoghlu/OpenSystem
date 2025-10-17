/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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

//===- Disassembler.h - Text File Disassembler ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the disassembler of strings of bytes written in
// hexadecimal, from standard input or from a file.
//
//===----------------------------------------------------------------------===//

#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H

#include <string>

namespace llvm {

class MemoryBuffer;
class Target;
class raw_ostream;
class SourceMgr;
class MCSubtargetInfo;
class MCStreamer;

class Disassembler {
public:
  static int disassemble(const Target &T,
                         const std::string &Triple,
                         MCSubtargetInfo &STI,
                         MCStreamer &Streamer,
                         MemoryBuffer &Buffer,
                         SourceMgr &SM,
                         raw_ostream &Out);

  static int disassembleEnhanced(const std::string &tripleString,
                                 MemoryBuffer &buffer,
                                 SourceMgr &SM,
                                 raw_ostream &Out);
};

} // namespace llvm

#endif
