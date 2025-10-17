/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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

//===- lib/MC/MCSectionCOFF.cpp - COFF Code Section Representation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCSectionCOFF::~MCSectionCOFF() {} // anchor.

// ShouldOmitSectionDirective - Decides whether a '.section' directive
// should be printed before the section name
bool MCSectionCOFF::ShouldOmitSectionDirective(StringRef Name,
                                               const MCAsmInfo &MAI) const {

  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (Name == ".text" || Name == ".data" || Name == ".bss")
    return true;

  return false;
}

void MCSectionCOFF::PrintSwitchToSection(const MCAsmInfo &MAI,
                                         raw_ostream &OS) const {

  // standard sections don't require the '.section'
  if (ShouldOmitSectionDirective(SectionName, MAI)) {
    OS << '\t' << getSectionName() << '\n';
    return;
  }

  OS << "\t.section\t" << getSectionName() << ",\"";
  if (getKind().isText())
    OS << 'x';
  if (getKind().isWriteable())
    OS << 'w';
  else
    OS << 'r';
  if (getCharacteristics() & COFF::IMAGE_SCN_MEM_DISCARDABLE)
    OS << 'n';
  OS << "\"\n";

  if (getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT) {
    switch (Selection) {
      case COFF::IMAGE_COMDAT_SELECT_NODUPLICATES:
        OS << "\t.linkonce one_only\n";
        break;
      case COFF::IMAGE_COMDAT_SELECT_ANY:
        OS << "\t.linkonce discard\n";
        break;
      case COFF::IMAGE_COMDAT_SELECT_SAME_SIZE:
        OS << "\t.linkonce same_size\n";
        break;
      case COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH:
        OS << "\t.linkonce same_contents\n";
        break;
    //NOTE: as of binutils 2.20, there is no way to specifiy select largest
    //      with the .linkonce directive. For now, we treat it as an invalid
    //      comdat selection value.
      case COFF::IMAGE_COMDAT_SELECT_LARGEST:
    //  OS << "\t.linkonce largest\n";
    //  break;
      default:
        assert (0 && "unsupported COFF selection type");
        break;
    }
  }
}

bool MCSectionCOFF::UseCodeAlign() const {
  return getKind().isText();
}

bool MCSectionCOFF::isVirtualSection() const {
  return getCharacteristics() & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
}
