/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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

//===-LTOCodeGenerator.h - LLVM Link Time Optimizer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LTOCodeGenerator class.
//
//===----------------------------------------------------------------------===//

#ifndef LTO_CODE_GENERATOR_H
#define LTO_CODE_GENERATOR_H

#include "llvm/Linker.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm-c/lto.h"
#include <string>

namespace llvm {
  class LLVMContext;
  class GlobalValue;
  class Mangler;
  class MemoryBuffer;
  class TargetMachine;
  class raw_ostream;
}

//===----------------------------------------------------------------------===//
/// LTOCodeGenerator - C++ class which implements the opaque lto_code_gen_t
/// type.
///
struct LTOCodeGenerator {
  static const char *getVersionString();

  LTOCodeGenerator();
  ~LTOCodeGenerator();

  bool addModule(struct LTOModule*, std::string &errMsg);
  bool setDebugInfo(lto_debug_model, std::string &errMsg);
  bool setCodePICModel(lto_codegen_model, std::string &errMsg);

  void setCpu(const char* mCpu) { _mCpu = mCpu; }

  void addMustPreserveSymbol(const char* sym) {
    _mustPreserveSymbols[sym] = 1;
  }

  bool writeMergedModules(const char *path, std::string &errMsg);
  bool compile_to_file(const char **name, std::string &errMsg);
  const void *compile(size_t *length, std::string &errMsg);
  void setCodeGenDebugOptions(const char *opts);

private:
  bool generateObjectFile(llvm::raw_ostream &out, std::string &errMsg);
  void applyScopeRestrictions();
  void applyRestriction(llvm::GlobalValue &GV,
                        std::vector<const char*> &mustPreserveList,
                        llvm::SmallPtrSet<llvm::GlobalValue*, 8> &asmUsed,
                        llvm::Mangler &mangler);
  bool determineTarget(std::string &errMsg);

  typedef llvm::StringMap<uint8_t> StringSet;

  llvm::LLVMContext&          _context;
  llvm::Linker                _linker;
  llvm::TargetMachine*        _target;
  bool                        _emitDwarfDebugInfo;
  bool                        _scopeRestrictionsDone;
  lto_codegen_model           _codeModel;
  StringSet                   _mustPreserveSymbols;
  StringSet                   _asmUndefinedRefs;
  llvm::MemoryBuffer*         _nativeObjectFile;
  std::vector<char*>          _codegenOptions;
  std::string                 _mCpu;
  std::string                 _nativeObjectPath;
};

#endif // LTO_CODE_GENERATOR_H
