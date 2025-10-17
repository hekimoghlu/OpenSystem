/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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

//===- NVPTXSection.h - NVPTX-specific section representation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTXSection class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_NVPTXSECTION_H
#define LLVM_NVPTXSECTION_H

#include "llvm/MC/MCSection.h"
#include "llvm/GlobalVariable.h"
#include <vector>

namespace llvm {
/// NVPTXSection - Represents a section in PTX
/// PTX does not have sections. We create this class in order to use
/// the ASMPrint interface.
///
class NVPTXSection : public MCSection {

public:
  NVPTXSection(SectionVariant V, SectionKind K) : MCSection(V, K) {}
  ~NVPTXSection() {}

  /// Override this as NVPTX has its own way of printing switching
  /// to a section.
  virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                    raw_ostream &OS) const {}

  /// Base address of PTX sections is zero.
  virtual bool isBaseAddressKnownZero() const { return true; }
  virtual bool UseCodeAlign() const { return false; }
  virtual bool isVirtualSection() const { return false; }
};

} // end namespace llvm

#endif
