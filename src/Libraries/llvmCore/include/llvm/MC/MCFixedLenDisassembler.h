/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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

//===-- llvm/MC/MCFixedLenDisassembler.h - Decoder driver -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Fixed length disassembler decoder state machine driver.
//===----------------------------------------------------------------------===//
#ifndef MCFIXEDLENDISASSEMBLER_H
#define MCFIXEDLENDISASSEMBLER_H

namespace llvm {

namespace MCD {
// Disassembler state machine opcodes.
enum DecoderOps {
  OPC_ExtractField = 1, // OPC_ExtractField(uint8_t Start, uint8_t Len)
  OPC_FilterValue,      // OPC_FilterValue(uleb128 Val, uint16_t NumToSkip)
  OPC_CheckField,       // OPC_CheckField(uint8_t Start, uint8_t Len,
                        //                uleb128 Val, uint16_t NumToSkip)
  OPC_CheckPredicate,   // OPC_CheckPredicate(uleb128 PIdx, uint16_t NumToSkip)
  OPC_Decode,           // OPC_Decode(uleb128 Opcode, uleb128 DIdx)
  OPC_SoftFail,         // OPC_SoftFail(uleb128 PMask, uleb128 NMask)
  OPC_Fail              // OPC_Fail()
};

} // namespace MCDecode
} // namespace llvm

#endif
