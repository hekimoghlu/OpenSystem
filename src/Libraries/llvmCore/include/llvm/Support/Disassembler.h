/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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

//===- llvm/Support/Disassembler.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the necessary glue to call external disassembler
// libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_DISASSEMBLER_H
#define LLVM_SYSTEM_DISASSEMBLER_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
namespace sys {

/// This function returns true, if there is possible to use some external
/// disassembler library. False otherwise.
bool hasDisassembler();

/// This function provides some "glue" code to call external disassembler
/// libraries.
std::string disassembleBuffer(uint8_t* start, size_t length, uint64_t pc = 0);

}
}

#endif // LLVM_SYSTEM_DISASSEMBLER_H
