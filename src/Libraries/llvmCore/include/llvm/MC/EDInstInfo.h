/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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

//===-- llvm/MC/EDInstInfo.h - EDis instruction info ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef EDINSTINFO_H
#define EDINSTINFO_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

#define EDIS_MAX_OPERANDS 13
#define EDIS_MAX_SYNTAXES 2

struct EDInstInfo {
  uint8_t       instructionType;
  uint8_t       numOperands;
  uint8_t       operandTypes[EDIS_MAX_OPERANDS];
  uint8_t       operandFlags[EDIS_MAX_OPERANDS];
  const signed char operandOrders[EDIS_MAX_SYNTAXES][EDIS_MAX_OPERANDS];
};

} // namespace llvm

#endif
