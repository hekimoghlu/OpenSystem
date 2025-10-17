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

//===-- llvm/ADT/SetOperations.h - Generic Set Operations -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines generic set operations that may be used on set's of
// different types, and different element types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SETOPERATIONS_H
#define LLVM_ADT_SETOPERATIONS_H

namespace llvm {

/// set_union(A, B) - Compute A := A u B, return whether A changed.
///
template <class S1Ty, class S2Ty>
bool set_union(S1Ty &S1, const S2Ty &S2) {
  bool Changed = false;

  for (typename S2Ty::const_iterator SI = S2.begin(), SE = S2.end();
       SI != SE; ++SI)
    if (S1.insert(*SI).second)
      Changed = true;

  return Changed;
}

/// set_intersect(A, B) - Compute A := A ^ B
/// Identical to set_intersection, except that it works on set<>'s and
/// is nicer to use.  Functionally, this iterates through S1, removing
/// elements that are not contained in S2.
///
template <class S1Ty, class S2Ty>
void set_intersect(S1Ty &S1, const S2Ty &S2) {
   for (typename S1Ty::iterator I = S1.begin(); I != S1.end();) {
     const typename S1Ty::key_type &E = *I;
     ++I;
     if (!S2.count(E)) S1.erase(E);   // Erase element if not in S2
   }
}

/// set_difference(A, B) - Return A - B
///
template <class S1Ty, class S2Ty>
S1Ty set_difference(const S1Ty &S1, const S2Ty &S2) {
  S1Ty Result;
  for (typename S1Ty::const_iterator SI = S1.begin(), SE = S1.end();
       SI != SE; ++SI)
    if (!S2.count(*SI))       // if the element is not in set2
      Result.insert(*SI);
  return Result;
}

/// set_subtract(A, B) - Compute A := A - B
///
template <class S1Ty, class S2Ty>
void set_subtract(S1Ty &S1, const S2Ty &S2) {
  for (typename S2Ty::const_iterator SI = S2.begin(), SE = S2.end();
       SI != SE; ++SI)
    S1.erase(*SI);
}

} // End llvm namespace

#endif
