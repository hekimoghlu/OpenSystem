/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include <wtf/StackCheck.h>

#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
#include <wtf/DataLog.h>
#endif

namespace WTF {

#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE

NO_RETURN_DUE_TO_CRASH void StackCheck::Scope::reportVerificationFailureAndCrash()
{
    uint8_t* currentStackCheckpoint = m_checker.m_lastStackCheckpoint;
    uint8_t* previousStackCheckpoint = m_savedLastStackCheckpoint;
    ptrdiff_t stackBetweenCheckpoints = previousStackCheckpoint - currentStackCheckpoint;

    dataLogLn("Stack check failure:");
    dataLogLn("    Previous checkpoint stack position: ", RawPointer(previousStackCheckpoint));
    dataLogLn("    Current checkpoint stack position: ", RawPointer(currentStackCheckpoint));
    dataLogLn("    Stack between checkpoints: ", stackBetweenCheckpoints);
    dataLogLn("    ReservedZone space: ", m_checker.m_reservedZone);
    dataLogLn();
    if constexpr (verboseStackCheckVerification) {
        dataLogLn("    Stack at previous checkpoint:");
        dataLogLn(StackTracePrinter { m_savedLastCheckpointStackTrace->stack(), "      " });
        dataLogLn("    Stack at current checkpoint:");
        dataLogLn(StackTracePrinter { m_checker.m_lastCheckpointStackTrace->stack(), "      " });
    } else {
        dataLogLn("    To see the stack traces at the 2 checkpoints, set verboseStackCheckVerification to true in StackCheck.h, rebuild, and re-run your test.");
        dataLogLn();
    }

    RELEASE_ASSERT(stackBetweenCheckpoints > 0);
    RELEASE_ASSERT(previousStackCheckpoint - currentStackCheckpoint < static_cast<ptrdiff_t>(m_checker.m_reservedZone));
    RELEASE_ASSERT_NOT_REACHED();
}

#endif // VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE

} // namespace WTF

