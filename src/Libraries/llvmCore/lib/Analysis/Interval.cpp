/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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

//===- Interval.cpp - Interval class code ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the Interval class, which represents a
// partition of a control flow graph of some kind.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Interval.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Interval Implementation
//===----------------------------------------------------------------------===//

// isLoop - Find out if there is a back edge in this interval...
//
bool Interval::isLoop() const {
  // There is a loop in this interval iff one of the predecessors of the header
  // node lives in the interval.
  for (::pred_iterator I = ::pred_begin(HeaderNode), E = ::pred_end(HeaderNode);
       I != E; ++I)
    if (contains(*I))
      return true;
  return false;
}


void Interval::print(raw_ostream &OS) const {
  OS << "-------------------------------------------------------------\n"
       << "Interval Contents:\n";

  // Print out all of the basic blocks in the interval...
  for (std::vector<BasicBlock*>::const_iterator I = Nodes.begin(),
         E = Nodes.end(); I != E; ++I)
    OS << **I << "\n";

  OS << "Interval Predecessors:\n";
  for (std::vector<BasicBlock*>::const_iterator I = Predecessors.begin(),
         E = Predecessors.end(); I != E; ++I)
    OS << **I << "\n";

  OS << "Interval Successors:\n";
  for (std::vector<BasicBlock*>::const_iterator I = Successors.begin(),
         E = Successors.end(); I != E; ++I)
    OS << **I << "\n";
}
