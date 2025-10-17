/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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

//===-- SPUMCAsmInfo.cpp - Cell SPU asm properties ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUMCAsmInfo.h"
using namespace llvm;

void SPULinuxMCAsmInfo::anchor() { }

SPULinuxMCAsmInfo::SPULinuxMCAsmInfo(const Target &T, StringRef TT) {
  IsLittleEndian = false;

  ZeroDirective = "\t.space\t";
  Data64bitsDirective = "\t.quad\t";
  AlignmentIsInBytes = false;
      
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";

  // Has leb128
  HasLEB128 = true;

  SupportsDebugInformation = true;

  // Exception handling is not supported on CellSPU (think about it: you only
  // have 256K for code+data. Would you support exception handling?)
  ExceptionsType = ExceptionHandling::None;

  // SPU assembly requires ".section" before ".bss" 
  UsesELFSectionDirectiveForBSS = true;  
}

