/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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

//===-- llvm/Assembly/Parser.h - Parser for VM assembly files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These classes are implemented by the lib/AsmParser library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PARSER_H
#define LLVM_ASSEMBLY_PARSER_H

#include <string>

namespace llvm {

class Module;
class MemoryBuffer;
class SMDiagnostic;
class LLVMContext;

/// This function is the main interface to the LLVM Assembly Parser. It parses
/// an ASCII file that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a file
Module *ParseAssemblyFile(
  const std::string &Filename, ///< The name of the file to parse
  SMDiagnostic &Error,         ///< Error result info.
  LLVMContext &Context         ///< Context in which to allocate globals info.
);

/// The function is a secondary interface to the LLVM Assembly Parser. It parses
/// an ASCII string that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a string
Module *ParseAssemblyString(
  const char *AsmString, ///< The string containing assembly
  Module *M,             ///< A module to add the assembly too.
  SMDiagnostic &Error,   ///< Error result info.
  LLVMContext &Context
);

/// This function is the low-level interface to the LLVM Assembly Parser.
/// ParseAssemblyFile and ParseAssemblyString are wrappers around this function.
/// @brief Parse LLVM Assembly from a MemoryBuffer. This function *always*
/// takes ownership of the MemoryBuffer.
Module *ParseAssembly(
    MemoryBuffer *F,     ///< The MemoryBuffer containing assembly
    Module *M,           ///< A module to add the assembly too.
    SMDiagnostic &Err,   ///< Error result info.
    LLVMContext &Context
);

} // End llvm namespace

#endif
