/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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

//===-- Solution.h ------- PBQP Solution ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PBQP Solution class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_SOLUTION_H
#define LLVM_CODEGEN_PBQP_SOLUTION_H

#include "Math.h"
#include "Graph.h"

#include <map>

namespace PBQP {

  /// \brief Represents a solution to a PBQP problem.
  ///
  /// To get the selection for each node in the problem use the getSelection method.
  class Solution {
  private:

    typedef std::map<Graph::ConstNodeItr, unsigned,
                     NodeItrComparator> SelectionsMap;
    SelectionsMap selections;

    unsigned r0Reductions, r1Reductions, r2Reductions, rNReductions;

  public:

    /// \brief Initialise an empty solution.
    Solution()
      : r0Reductions(0), r1Reductions(0), r2Reductions(0), rNReductions(0) {}

    /// \brief Number of nodes for which selections have been made.
    /// @return Number of nodes for which selections have been made.
    unsigned numNodes() const { return selections.size(); }

    /// \brief Records a reduction via the R0 rule. Should be called from the
    ///        solver only.
    void recordR0() { ++r0Reductions; }

    /// \brief Returns the number of R0 reductions applied to solve the problem.
    unsigned numR0Reductions() const { return r0Reductions; }

    /// \brief Records a reduction via the R1 rule. Should be called from the
    ///        solver only.
    void recordR1() { ++r1Reductions; }

    /// \brief Returns the number of R1 reductions applied to solve the problem.
    unsigned numR1Reductions() const { return r1Reductions; }

    /// \brief Records a reduction via the R2 rule. Should be called from the
    ///        solver only.
    void recordR2() { ++r2Reductions; }

    /// \brief Returns the number of R2 reductions applied to solve the problem.
    unsigned numR2Reductions() const { return r2Reductions; }

    /// \brief Records a reduction via the RN rule. Should be called from the
    ///        solver only.
    void recordRN() { ++ rNReductions; }

    /// \brief Returns the number of RN reductions applied to solve the problem.
    unsigned numRNReductions() const { return rNReductions; }

    /// \brief Set the selection for a given node.
    /// @param nItr Node iterator.
    /// @param selection Selection for nItr.
    void setSelection(Graph::NodeItr nItr, unsigned selection) {
      selections[nItr] = selection;
    }

    /// \brief Get a node's selection.
    /// @param nItr Node iterator.
    /// @return The selection for nItr;
    unsigned getSelection(Graph::ConstNodeItr nItr) const {
      SelectionsMap::const_iterator sItr = selections.find(nItr);
      assert(sItr != selections.end() && "No selection for node.");
      return sItr->second;
    }

  };

}

#endif // LLVM_CODEGEN_PBQP_SOLUTION_H
