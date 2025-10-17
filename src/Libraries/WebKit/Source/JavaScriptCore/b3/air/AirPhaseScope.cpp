/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "AirPhaseScope.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirValidate.h"
#include "B3Common.h"
#include <wtf/StringPrintStream.h>

namespace JSC { namespace B3 { namespace Air {

PhaseScope::PhaseScope(Code& code, ASCIILiteral name)
    : m_code(code)
    , m_name(name)
    , m_timingScope("Air"_s, name)
{
    if (shouldDumpIRAtEachPhase(AirMode)) {
        dataLog("Air after ", code.lastPhaseName(), ", before ", name, ":\n");
        dataLog(code);
    }

    if (shouldSaveIRBeforePhase())
        m_dumpBefore = toCString(code);
}

PhaseScope::~PhaseScope()
{
    m_code.setLastPhaseName(m_name);
    if (shouldValidateIRAtEachPhase())
        validate(m_code, m_dumpBefore.data());
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
