/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

//===-- AssemblyAnnotationWriter.h - Annotation .ll files -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Clients of the assembly writer can use this interface to add their own
// special-purpose annotations to LLVM assembly language printouts.  Note that
// the assembly parser won't be able to parse these, in general, so
// implementations are advised to print stuff as LLVM comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_ASMANNOTATIONWRITER_H
#define LLVM_ASSEMBLY_ASMANNOTATIONWRITER_H

namespace llvm {

class Function;
class BasicBlock;
class Instruction;
class Value;
class formatted_raw_ostream;

class AssemblyAnnotationWriter {
public:

  virtual ~AssemblyAnnotationWriter();

  /// emitFunctionAnnot - This may be implemented to emit a string right before
  /// the start of a function.
  virtual void emitFunctionAnnot(const Function *,
                                 formatted_raw_ostream &) {}

  /// emitBasicBlockStartAnnot - This may be implemented to emit a string right
  /// after the basic block label, but before the first instruction in the
  /// block.
  virtual void emitBasicBlockStartAnnot(const BasicBlock *,
                                        formatted_raw_ostream &) {
  }

  /// emitBasicBlockEndAnnot - This may be implemented to emit a string right
  /// after the basic block.
  virtual void emitBasicBlockEndAnnot(const BasicBlock *,
                                      formatted_raw_ostream &) {
  }

  /// emitInstructionAnnot - This may be implemented to emit a string right
  /// before an instruction is emitted.
  virtual void emitInstructionAnnot(const Instruction *, 
                                    formatted_raw_ostream &) {}

  /// printInfoComment - This may be implemented to emit a comment to the
  /// right of an instruction or global value.
  virtual void printInfoComment(const Value &, formatted_raw_ostream &) {}
};

} // End llvm namespace

#endif
