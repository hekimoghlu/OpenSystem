/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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

//===- ValueTypes.cpp - Tablegen extended ValueType implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The EVT type is used by tablegen as well as in LLVM. In order to handle
// extended types, the EVT type uses support functions that call into
// LLVM's type system code. These aren't accessible in tablegen, so this
// file provides simple replacements.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ValueTypes.h"
#include <map>
using namespace llvm;

namespace llvm {

class Type {
public:
  virtual unsigned getSizeInBits() const = 0;
  virtual ~Type() {}
};

}

class ExtendedIntegerType : public Type {
  unsigned BitWidth;
public:
  explicit ExtendedIntegerType(unsigned bits)
    : BitWidth(bits) {}
  unsigned getSizeInBits() const {
    return getBitWidth();
  }
  unsigned getBitWidth() const {
    return BitWidth;
  }
};

class ExtendedVectorType : public Type {
  EVT ElementType;
  unsigned NumElements;
public:
  ExtendedVectorType(EVT elty, unsigned num)
    : ElementType(elty), NumElements(num) {}
  unsigned getSizeInBits() const {
    return getNumElements() * getElementType().getSizeInBits();
  }
  EVT getElementType() const {
    return ElementType;
  }
  unsigned getNumElements() const {
    return NumElements;
  }
};

static std::map<unsigned, const Type *>
  ExtendedIntegerTypeMap;
static std::map<std::pair<uintptr_t, uintptr_t>, const Type *>
  ExtendedVectorTypeMap;

bool EVT::isExtendedFloatingPoint() const {
  assert(isExtended() && "Type is not extended!");
  // Extended floating-point types are not supported yet.
  return false;
}

bool EVT::isExtendedInteger() const {
  assert(isExtended() && "Type is not extended!");
  return dynamic_cast<const ExtendedIntegerType *>(LLVMTy) != 0;
}

bool EVT::isExtendedVector() const {
  assert(isExtended() && "Type is not extended!");
  return dynamic_cast<const ExtendedVectorType *>(LLVMTy) != 0;
}

bool EVT::isExtended64BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 64;
}

bool EVT::isExtended128BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 128;
}

EVT EVT::getExtendedVectorElementType() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getElementType();
}

unsigned EVT::getExtendedVectorNumElements() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getNumElements();
}

unsigned EVT::getExtendedSizeInBits() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->getSizeInBits();
}
