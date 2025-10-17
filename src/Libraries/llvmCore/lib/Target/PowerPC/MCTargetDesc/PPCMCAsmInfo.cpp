/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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

//===-- PPCMCAsmInfo.cpp - PPC asm properties -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MCAsmInfoDarwin properties.
//
//===----------------------------------------------------------------------===//

#include "PPCMCAsmInfo.h"
using namespace llvm;

void PPCMCAsmInfoDarwin::anchor() { }

PPCMCAsmInfoDarwin::PPCMCAsmInfoDarwin(bool is64Bit) {
  if (is64Bit)
    PointerSize = 8;
  IsLittleEndian = false;

  PCSymbol = ".";
  CommentString = ";";
  ExceptionsType = ExceptionHandling::DwarfCFI;

  if (!is64Bit)
    Data64bitsDirective = 0;      // We can't emit a 64-bit unit in PPC32 mode.

  AssemblerDialect = 1;           // New-Style mnemonics.
  SupportsDebugInformation= true; // Debug information.
}

void PPCLinuxMCAsmInfo::anchor() { }

PPCLinuxMCAsmInfo::PPCLinuxMCAsmInfo(bool is64Bit) {
  if (is64Bit)
    PointerSize = 8;
  IsLittleEndian = false;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  
  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;  

  // Debug Information
  SupportsDebugInformation = true;

  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;
    
  ZeroDirective = "\t.space\t";
  Data64bitsDirective = is64Bit ? "\t.quad\t" : 0;
  AssemblerDialect = 0;           // Old-Style mnemonics.
}

