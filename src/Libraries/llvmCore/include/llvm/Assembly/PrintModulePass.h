/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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

//===- llvm/Assembly/PrintModulePass.h - Printing Pass ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two passes to print out a module.  The PrintModulePass pass
// simply prints out the entire module when it is executed.  The
// PrintFunctionPass class is designed to be pipelined with other
// FunctionPass's, and prints out the functions of the module as they are
// processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PRINTMODULEPASS_H
#define LLVM_ASSEMBLY_PRINTMODULEPASS_H

#include <string>

namespace llvm {
  class FunctionPass;
  class ModulePass;
  class raw_ostream;
  
  /// createPrintModulePass - Create and return a pass that writes the
  /// module to the specified raw_ostream.
  ModulePass *createPrintModulePass(raw_ostream *OS,
                                    bool DeleteStream=false,
                                    const std::string &Banner = "");
  
  /// createPrintFunctionPass - Create and return a pass that prints
  /// functions to the specified raw_ostream as they are processed.
  FunctionPass *createPrintFunctionPass(const std::string &Banner,
                                        raw_ostream *OS, 
                                        bool DeleteStream=false);  

} // End llvm namespace

#endif
