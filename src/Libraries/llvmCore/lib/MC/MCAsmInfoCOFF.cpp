/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

//===-- MCAsmInfoCOFF.cpp - COFF asm properties -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on COFF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoCOFF.h"
using namespace llvm;

void MCAsmInfoCOFF::anchor() { }

MCAsmInfoCOFF::MCAsmInfoCOFF() {
  GlobalPrefix = "_";
  // MingW 4.5 and later support .comm with log2 alignment, but .lcomm uses byte
  // alignment.
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  LinkOnceDirective = "\t.linkonce discard\n";

  // Doesn't support visibility:
  HiddenVisibilityAttr = HiddenDeclarationVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  SupportsDebugInformation = true;
  DwarfSectionOffsetDirective = "\t.secrel32\t";
  HasMicrosoftFastStdCallMangling = true;
}

void MCAsmInfoMicrosoft::anchor() { }

MCAsmInfoMicrosoft::MCAsmInfoMicrosoft() {
  AllowQuotesInName = true;
}

void MCAsmInfoGNUCOFF::anchor() { }

MCAsmInfoGNUCOFF::MCAsmInfoGNUCOFF() {

}
