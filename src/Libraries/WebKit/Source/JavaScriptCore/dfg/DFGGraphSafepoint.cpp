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
#include "config.h"
#include "DFGGraphSafepoint.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

GraphSafepoint::GraphSafepoint(Graph& graph, Safepoint::Result& result, bool keepDependencesLive)
    : m_safepoint(graph.m_plan, result)
{
    m_safepoint.add(&graph);
    m_safepoint.begin(keepDependencesLive);
}

GraphSafepoint::~GraphSafepoint() = default;

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

