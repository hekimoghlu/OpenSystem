/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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

//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// CallDAG.h: Defines a call graph DAG of functions to be re-used accross
// analyses, allows to efficiently traverse the functions in topological
// order.

#ifndef COMPILER_TRANSLATOR_CALLDAG_H_
#define COMPILER_TRANSLATOR_CALLDAG_H_

#include <map>

#include "compiler/translator/IntermNode.h"

namespace sh
{

// The translator needs to analyze the the graph of the function calls
// to run checks and analyses; since in GLSL recursion is not allowed
// that graph is a DAG.
// This class is used to precompute that function call DAG so that it
// can be reused by multiple analyses.
//
// It stores a vector of function records, with one record per defined function.
// Records are accessed by index but a function symbol id can be converted
// to the index of the corresponding record. The records contain the AST node
// of the function definition and the indices of the function's callees.
//
// In addition, records are in reverse topological order: a function F being
// called by a function G will have index index(F) < index(G), that way
// depth-first analysis becomes analysis in the order of indices.

class CallDAG : angle::NonCopyable
{
  public:
    CallDAG();
    ~CallDAG();

    struct Record
    {
        TIntermFunctionDefinition *node;  // Guaranteed to be non-null.
        std::vector<int> callees;
    };

    enum InitResult
    {
        INITDAG_SUCCESS,
        INITDAG_RECURSION,
        INITDAG_UNDEFINED,
    };

    // Returns INITDAG_SUCCESS if it was able to create the DAG, otherwise prints
    // the initialization error in diagnostics, if present.
    InitResult init(TIntermNode *root, TDiagnostics *diagnostics);

    // Returns InvalidIndex if the function wasn't found
    size_t findIndex(const TSymbolUniqueId &id) const;

    const Record &getRecordFromIndex(size_t index) const;
    size_t size() const;
    void clear();

    const static size_t InvalidIndex;

  private:
    std::vector<Record> mRecords;
    std::map<int, int> mFunctionIdToIndex;

    class CallDAGCreator;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_CALLDAG_H_
