/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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

//===-- MCAsmBackend.cpp - Target MC Assembly Backend ----------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCFixupKindInfo.h"
using namespace llvm;

MCAsmBackend::MCAsmBackend()
  : HasReliableSymbolDifference(false), HasDataInCodeSupport(false) {}

MCAsmBackend::~MCAsmBackend() {}

const MCFixupKindInfo &
MCAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  static const MCFixupKindInfo Builtins[] = {
    { "FK_Data_1",  0,  8, 0 },
    { "FK_Data_2",  0, 16, 0 },
    { "FK_Data_4",  0, 32, 0 },
    { "FK_Data_8",  0, 64, 0 },
    { "FK_PCRel_1", 0,  8, MCFixupKindInfo::FKF_IsPCRel },
    { "FK_PCRel_2", 0, 16, MCFixupKindInfo::FKF_IsPCRel },
    { "FK_PCRel_4", 0, 32, MCFixupKindInfo::FKF_IsPCRel },
    { "FK_PCRel_8", 0, 64, MCFixupKindInfo::FKF_IsPCRel },
    { "FK_GPRel_1", 0,  8, 0 },
    { "FK_GPRel_2", 0, 16, 0 },
    { "FK_GPRel_4", 0, 32, 0 },
    { "FK_GPRel_8", 0, 64, 0 },
    { "FK_SecRel_1", 0,  8, 0 },
    { "FK_SecRel_2", 0, 16, 0 },
    { "FK_SecRel_4", 0, 32, 0 },
    { "FK_SecRel_8", 0, 64, 0 }
  };

  assert((size_t)Kind <= sizeof(Builtins) / sizeof(Builtins[0]) &&
         "Unknown fixup kind");
  return Builtins[Kind];
}
