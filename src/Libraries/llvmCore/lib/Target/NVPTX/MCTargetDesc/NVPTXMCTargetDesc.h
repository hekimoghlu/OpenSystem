/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

//===-- NVPTXMCTargetDesc.h - NVPTX Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides NVPTX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXMCTARGETDESC_H
#define NVPTXMCTARGETDESC_H

namespace llvm {
class Target;

extern Target TheNVPTXTarget32;
extern Target TheNVPTXTarget64;

} // End llvm namespace

// Defines symbolic names for PTX registers.
#define GET_REGINFO_ENUM
#include "NVPTXGenRegisterInfo.inc"

// Defines symbolic names for the PTX instructions.
#define GET_INSTRINFO_ENUM
#include "NVPTXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "NVPTXGenSubtargetInfo.inc"

#endif
