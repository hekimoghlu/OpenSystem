/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "DFGPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGValidate.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

void Phase::validate()
{
    DFG::validate(m_graph, DumpGraph, m_graphDumpBeforePhase);
}

void Phase::beginPhase()
{
    if (Options::verboseValidationFailure()) {
        StringPrintStream out;
        m_graph.dump(out);
        m_graphDumpBeforePhase = out.toCString();
    }

    if (!shouldDumpGraphAtEachPhase(m_graph.m_plan.mode()))
        return;
    
    dataLog(m_graph.prefix(), "Beginning DFG phase ", m_name, ".\n");
    dataLog(m_graph.prefix(), "Before ", m_name, ":\n");
    m_graph.dump();
}

void Phase::endPhase()
{
    if (!Options::validateGraphAtEachPhase() || m_disableGraphValidation)
        return;
    validate();
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
