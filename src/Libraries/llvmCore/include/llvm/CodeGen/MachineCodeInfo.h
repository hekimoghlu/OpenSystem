/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

//===-- MachineCodeInfo.h - Class used to report JIT info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines MachineCodeInfo, a class used by the JIT ExecutionEngine
// to report information about the generated machine code.
//
// See JIT::runJITOnFunction for usage.
//
//===----------------------------------------------------------------------===//

#ifndef EE_MACHINE_CODE_INFO_H
#define EE_MACHINE_CODE_INFO_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class MachineCodeInfo {
private:
  size_t Size;   // Number of bytes in memory used
  void *Address; // The address of the function in memory

public:
  MachineCodeInfo() : Size(0), Address(0) {}

  void setSize(size_t s) {
    Size = s;
  }

  void setAddress(void *a) {
    Address = a;
  }

  size_t size() const {
    return Size;
  }

  void *address() const {
    return Address;
  }

};

}

#endif

