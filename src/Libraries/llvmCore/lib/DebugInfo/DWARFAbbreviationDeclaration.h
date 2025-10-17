/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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

//===-- DWARFAbbreviationDeclaration.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFABBREVIATIONDECLARATION_H
#define LLVM_DEBUGINFO_DWARFABBREVIATIONDECLARATION_H

#include "DWARFAttribute.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

class raw_ostream;

class DWARFAbbreviationDeclaration {
  uint32_t Code;
  uint32_t Tag;
  bool HasChildren;
  SmallVector<DWARFAttribute, 8> Attributes;
public:
  enum { InvalidCode = 0 };
  DWARFAbbreviationDeclaration()
    : Code(InvalidCode), Tag(0), HasChildren(0) {}

  uint32_t getCode() const { return Code; }
  uint32_t getTag() const { return Tag; }
  bool hasChildren() const { return HasChildren; }
  uint32_t getNumAttributes() const { return Attributes.size(); }
  uint16_t getAttrByIndex(uint32_t idx) const {
    return Attributes.size() > idx ? Attributes[idx].getAttribute() : 0;
  }
  uint16_t getFormByIndex(uint32_t idx) const {
    return Attributes.size() > idx ? Attributes[idx].getForm() : 0;
  }

  uint32_t findAttributeIndex(uint16_t attr) const;
  bool extract(DataExtractor data, uint32_t* offset_ptr);
  bool extract(DataExtractor data, uint32_t* offset_ptr, uint32_t code);
  bool isValid() const { return Code != 0 && Tag != 0; }
  void dump(raw_ostream &OS) const;
  const SmallVectorImpl<DWARFAttribute> &getAttributes() const {
    return Attributes;
  }
};

}

#endif
