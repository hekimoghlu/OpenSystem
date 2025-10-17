/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

//===--- DeltaAlgorithm.cpp - A Set Minimization Algorithm -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DeltaAlgorithm.h"
#include <algorithm>
#include <iterator>
using namespace llvm;

DeltaAlgorithm::~DeltaAlgorithm() {
}

bool DeltaAlgorithm::GetTestResult(const changeset_ty &Changes) {
  if (FailedTestsCache.count(Changes))
    return false;

  bool Result = ExecuteOneTest(Changes);
  if (!Result)
    FailedTestsCache.insert(Changes);

  return Result;
}

void DeltaAlgorithm::Split(const changeset_ty &S, changesetlist_ty &Res) {
  // FIXME: Allow clients to provide heuristics for improved splitting.

  // FIXME: This is really slow.
  changeset_ty LHS, RHS;
  unsigned idx = 0, N = S.size() / 2;
  for (changeset_ty::const_iterator it = S.begin(),
         ie = S.end(); it != ie; ++it, ++idx)
    ((idx < N) ? LHS : RHS).insert(*it);
  if (!LHS.empty())
    Res.push_back(LHS);
  if (!RHS.empty())
    Res.push_back(RHS);
}

DeltaAlgorithm::changeset_ty
DeltaAlgorithm::Delta(const changeset_ty &Changes,
                      const changesetlist_ty &Sets) {
  // Invariant: union(Res) == Changes
  UpdatedSearchState(Changes, Sets);

  // If there is nothing left we can remove, we are done.
  if (Sets.size() <= 1)
    return Changes;

  // Look for a passing subset.
  changeset_ty Res;
  if (Search(Changes, Sets, Res))
    return Res;

  // Otherwise, partition the sets if possible; if not we are done.
  changesetlist_ty SplitSets;
  for (changesetlist_ty::const_iterator it = Sets.begin(),
         ie = Sets.end(); it != ie; ++it)
    Split(*it, SplitSets);
  if (SplitSets.size() == Sets.size())
    return Changes;

  return Delta(Changes, SplitSets);
}

bool DeltaAlgorithm::Search(const changeset_ty &Changes,
                            const changesetlist_ty &Sets,
                            changeset_ty &Res) {
  // FIXME: Parallelize.
  for (changesetlist_ty::const_iterator it = Sets.begin(),
         ie = Sets.end(); it != ie; ++it) {
    // If the test passes on this subset alone, recurse.
    if (GetTestResult(*it)) {
      changesetlist_ty Sets;
      Split(*it, Sets);
      Res = Delta(*it, Sets);
      return true;
    }

    // Otherwise, if we have more than two sets, see if test passes on the
    // complement.
    if (Sets.size() > 2) {
      // FIXME: This is really slow.
      changeset_ty Complement;
      std::set_difference(
        Changes.begin(), Changes.end(), it->begin(), it->end(),
        std::insert_iterator<changeset_ty>(Complement, Complement.begin()));
      if (GetTestResult(Complement)) {
        changesetlist_ty ComplementSets;
        ComplementSets.insert(ComplementSets.end(), Sets.begin(), it);
        ComplementSets.insert(ComplementSets.end(), it + 1, Sets.end());
        Res = Delta(Complement, ComplementSets);
        return true;
      }
    }
  }

  return false;
}

DeltaAlgorithm::changeset_ty DeltaAlgorithm::Run(const changeset_ty &Changes) {
  // Check empty set first to quickly find poor test functions.
  if (GetTestResult(changeset_ty()))
    return changeset_ty();

  // Otherwise run the real delta algorithm.
  changesetlist_ty Sets;
  Split(Changes, Sets);

  return Delta(Changes, Sets);
}
