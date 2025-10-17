/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#pragma once

#include <wtf/StackBounds.h>
#include <wtf/Threading.h>

// We only enable the reserved zone size check by default on ASSERT_ENABLED
// builds (which usually mean Debug builds). However, it is more valuable to
// run this test on Release builds. That said, we don't want to do pay this
// cost always because we may need to do stack checks on hot paths too.
// Note also that we're deliberately using RELEASE_ASSERTs if the verification
// is turned on. This makes it easier for us to turn this on for Release builds
// for a reserved zone size calibration test runs.

#define VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE ASSERT_ENABLED

#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
#include <wtf/StackTrace.h>

constexpr bool verboseStackCheckVerification = false;
#endif

namespace WTF {

class StackCheck {
    WTF_MAKE_FAST_ALLOCATED;
public:
    class Scope {
    public:
        Scope(StackCheck& checker)
            : m_checker(checker)
        {
#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
            RELEASE_ASSERT(checker.m_ownerThread == &Thread::current());

            // Make sure that the stack interval between checks never exceed the
            // reservedZone size. If the interval exceeds the reservedZone size,
            // then we either need to:
            // 1. break the interval into smaller pieces (i.e. insert more checks), or
            // 2. use a larger reservedZone size.
            uint8_t* currentStackCheckpoint = static_cast<uint8_t*>(currentStackPointer());
            uint8_t* previousStackCheckpoint = m_checker.m_lastStackCheckpoint;
            ptrdiff_t stackBetweenCheckpoints = previousStackCheckpoint - currentStackCheckpoint;

            m_savedLastStackCheckpoint = m_checker.m_lastStackCheckpoint;
            m_checker.m_lastStackCheckpoint = currentStackCheckpoint;
            if constexpr (verboseStackCheckVerification) {
                m_savedLastCheckpointStackTrace = WTFMove(m_checker.m_lastCheckpointStackTrace);
                m_checker.m_lastCheckpointStackTrace = StackTrace::captureStackTrace(INT_MAX);
            }

            if (UNLIKELY(stackBetweenCheckpoints <= 0 || stackBetweenCheckpoints >= static_cast<ptrdiff_t>(m_checker.m_reservedZone)))
                reportVerificationFailureAndCrash();
#endif
        }

#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
        ~Scope()
        {
            m_checker.m_lastStackCheckpoint = m_savedLastStackCheckpoint;
            if constexpr (verboseStackCheckVerification)
                m_checker.m_lastCheckpointStackTrace = WTFMove(m_savedLastCheckpointStackTrace);
        }
#endif

        bool isSafeToRecurse() { return m_checker.isSafeToRecurse(); }

    private:
#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
        WTF_EXPORT_PRIVATE NO_RETURN_DUE_TO_CRASH void reportVerificationFailureAndCrash();
#endif

        StackCheck& m_checker;
#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
        uint8_t* m_savedLastStackCheckpoint;
        std::unique_ptr<StackTrace> m_savedLastCheckpointStackTrace;
#endif
    };

    StackCheck(const StackBounds& bounds = Thread::current().stack(), size_t minReservedZone = StackBounds::DefaultReservedZone)
        : m_stackLimit(bounds.recursionLimit(minReservedZone))
#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
        , m_ownerThread(&Thread::current())
        , m_lastStackCheckpoint(static_cast<uint8_t*>(currentStackPointer()))
        , m_reservedZone(minReservedZone)
#endif
    { }

    inline bool isSafeToRecurse() { return currentStackPointer() >= m_stackLimit; }

private:
    void* m_stackLimit;
#if VERIFY_STACK_CHECK_RESERVED_ZONE_SIZE
    Thread* m_ownerThread;
    uint8_t* m_lastStackCheckpoint;
    size_t m_reservedZone;
    std::unique_ptr<StackTrace> m_lastCheckpointStackTrace;
#endif

    friend class Scope;
};

} // namespace WTF

using WTF::StackCheck;
