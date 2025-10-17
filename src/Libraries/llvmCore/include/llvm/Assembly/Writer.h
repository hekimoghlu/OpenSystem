/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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

//===-- llvm/Assembly/Writer.h - Printer for LLVM assembly files --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This functionality is implemented by lib/VMCore/AsmWriter.cpp.
// This library is used to print LLVM assembly language files to an iostream. It
// can print LLVM code at a variety of granularities, including Modules,
// BasicBlocks, and Instructions.  This makes it useful for debugging.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_WRITER_H
#define LLVM_ASSEMBLY_WRITER_H

namespace llvm {

class Module;
class Value;
class raw_ostream;

// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.  If you specify a Module for context,
// then even constants get pretty-printed; for example, the type of a null
// pointer is printed symbolically.
//
void WriteAsOperand(raw_ostream &, const Value *, bool PrintTy = true,
                    const Module *Context = 0);

} // End llvm namespace

#endif
