/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#include "B3PhaseScope.h"

#if ENABLE(B3_JIT)

#include "B3Common.h"
#include "B3Procedure.h"
#include "B3Validate.h"
#include <wtf/DataLog.h>
#include <wtf/StringPrintStream.h>

namespace JSC { namespace B3 {

PhaseScope::PhaseScope(Procedure& procedure, ASCIILiteral name)
    : m_procedure(procedure)
    , m_name(name)
    , m_timingScope("B3"_s, name)
{
    if (shouldDumpIRAtEachPhase(B3Mode)) {
        dataLog("B3 after ", procedure.lastPhaseName(), ", before ", name, ":\n");
        dataLog(procedure);
    }

    if (shouldSaveIRBeforePhase())
        m_dumpBefore = toCString(procedure);
}

PhaseScope::~PhaseScope()
{
    m_procedure.setLastPhaseName(m_name);
    if (shouldValidateIRAtEachPhase())
        validate(m_procedure, m_dumpBefore.data());
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
