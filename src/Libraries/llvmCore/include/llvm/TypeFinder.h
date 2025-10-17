/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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

//===-- llvm/TypeFinder.h - Class for finding used struct types -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the TypeFinder class. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TYPEFINDER_H
#define LLVM_TYPEFINDER_H

#include "llvm/ADT/DenseSet.h"
#include <vector>

namespace llvm {

class MDNode;
class Module;
class StructType;
class Type;
class Value;

/// TypeFinder - Walk over a module, identifying all of the types that are
/// used by the module.
class TypeFinder {
  // To avoid walking constant expressions multiple times and other IR
  // objects, we keep several helper maps.
  DenseSet<const Value*> VisitedConstants;
  DenseSet<Type*> VisitedTypes;

  std::vector<StructType*> StructTypes;
  bool OnlyNamed;

public:
  TypeFinder() : OnlyNamed(false) {}

  void run(const Module &M, bool onlyNamed);
  void clear();

  typedef std::vector<StructType*>::iterator iterator;
  typedef std::vector<StructType*>::const_iterator const_iterator;

  iterator begin() { return StructTypes.begin(); }
  iterator end() { return StructTypes.end(); }

  const_iterator begin() const { return StructTypes.begin(); }
  const_iterator end() const { return StructTypes.end(); }

  bool empty() const { return StructTypes.empty(); }
  size_t size() const { return StructTypes.size(); }
  iterator erase(iterator I, iterator E) { return StructTypes.erase(I, E); }

  StructType *&operator[](unsigned Idx) { return StructTypes[Idx]; }

private:
  /// incorporateType - This method adds the type to the list of used
  /// structures if it's not in there already.
  void incorporateType(Type *Ty);

  /// incorporateValue - This method is used to walk operand lists finding types
  /// hiding in constant expressions and other operands that won't be walked in
  /// other ways.  GlobalValues, basic blocks, instructions, and inst operands
  /// are all explicitly enumerated.
  void incorporateValue(const Value *V);

  /// incorporateMDNode - This method is used to walk the operands of an MDNode
  /// to find types hiding within.
  void incorporateMDNode(const MDNode *V);
};

} // end llvm namespace

#endif
