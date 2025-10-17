/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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

//===-- llvm/Support/Win64EH.h ---Win64 EH Constants-------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains constants and structures used for implementing
// exception handling on Win64 platforms. For more information, see
// http://msdn.microsoft.com/en-us/library/1eyas8tf.aspx
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_WIN64EH_H
#define LLVM_SUPPORT_WIN64EH_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
namespace Win64EH {

/// UnwindOpcodes - Enumeration whose values specify a single operation in
/// the prolog of a function.
enum UnwindOpcodes {
  UOP_PushNonVol = 0,
  UOP_AllocLarge,
  UOP_AllocSmall,
  UOP_SetFPReg,
  UOP_SaveNonVol,
  UOP_SaveNonVolBig,
  UOP_SaveXMM128 = 8,
  UOP_SaveXMM128Big,
  UOP_PushMachFrame
};

/// UnwindCode - This union describes a single operation in a function prolog,
/// or part thereof.
union UnwindCode {
  struct {
    uint8_t codeOffset;
    uint8_t unwindOp:4,
            opInfo:4;
  } u;
  uint16_t frameOffset;
};

enum {
  /// UNW_ExceptionHandler - Specifies that this function has an exception
  /// handler.
  UNW_ExceptionHandler = 0x01,
  /// UNW_TerminateHandler - Specifies that this function has a termination
  /// handler.
  UNW_TerminateHandler = 0x02,
  /// UNW_ChainInfo - Specifies that this UnwindInfo structure is chained to
  /// another one.
  UNW_ChainInfo = 0x04
};

/// RuntimeFunction - An entry in the table of functions with unwind info.
struct RuntimeFunction {
  uint64_t startAddress;
  uint64_t endAddress;
  uint64_t unwindInfoOffset;
};

/// UnwindInfo - An entry in the exception table.
struct UnwindInfo {
  uint8_t version:3,
          flags:5;
  uint8_t prologSize;
  uint8_t numCodes;
  uint8_t frameRegister:4,
          frameOffset:4;
  UnwindCode unwindCodes[1];

  void *getLanguageSpecificData() {
    return reinterpret_cast<void *>(&unwindCodes[(numCodes+1) & ~1]);
  }
  uint64_t getLanguageSpecificHandlerOffset() {
    return *reinterpret_cast<uint64_t *>(getLanguageSpecificData());
  }
  void setLanguageSpecificHandlerOffset(uint64_t offset) {
    *reinterpret_cast<uint64_t *>(getLanguageSpecificData()) = offset;
  }
  RuntimeFunction *getChainedFunctionEntry() {
    return reinterpret_cast<RuntimeFunction *>(getLanguageSpecificData());
  }
  void *getExceptionData() {
    return reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(
                                                  getLanguageSpecificData())+1);
  }
};


} // End of namespace Win64EH
} // End of namespace llvm

#endif
