/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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

//===-- MCAsmInfoDarwin.cpp - Darwin asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoDarwin.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
using namespace llvm;

void MCAsmInfoDarwin::anchor() { }

MCAsmInfoDarwin::MCAsmInfoDarwin() {
  // Common settings for all Darwin targets.
  // Syntax:
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  LinkerPrivateGlobalPrefix = "l";
  AllowQuotesInName = true;
  HasSingleParameterDotFile = false;
  HasSubsectionsViaSymbols = true;

  AlignmentIsInBytes = false;
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::Log2Alignment;
  InlineAsmStart = " InlineAsm Start";
  InlineAsmEnd = " InlineAsm End";

  // Directives:
  WeakDefDirective = "\t.weak_definition ";
  WeakRefDirective = "\t.weak_reference ";
  ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
  HasMachoZeroFillDirective = true;  // Uses .zerofill
  HasMachoTBSSDirective = true; // Uses .tbss
  HasStaticCtorDtorReferenceInStaticMode = true;

  // FIXME: Darwin 10 and newer don't need this.
  LinkerRequiresNonEmptyDwarfLines = true;

  // FIXME: Change this once MC is the system assembler.
  HasAggressiveSymbolFolding = false;

  HiddenVisibilityAttr = MCSA_PrivateExtern;
  HiddenDeclarationVisibilityAttr = MCSA_Invalid;

  // Doesn't support protected visibility.
  ProtectedVisibilityAttr = MCSA_Invalid;

  HasDotTypeDotSizeDirective = false;
  HasNoDeadStrip = true;
  HasSymbolResolver = true;

  DwarfUsesRelocationsAcrossSections = false;
}
