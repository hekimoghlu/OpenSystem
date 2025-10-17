/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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

#include "ASTStringDumper.h"
#include <wtf/DataLog.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Seconds.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WGSL {

class ShaderModule;

static constexpr bool dumpASTBeforeEachPass = false;
static constexpr bool dumpASTAfterParsing = false;
static constexpr bool dumpASTAtEnd = false;
static constexpr bool alwaysDumpPassFailures = false;
static constexpr bool dumpPassFailure = dumpASTBeforeEachPass || dumpASTAfterParsing || dumpASTAtEnd || alwaysDumpPassFailures;
static constexpr bool dumpPhaseTimes = false;

static inline bool dumpASTIfNeeded(bool shouldDump, ShaderModule& program, const char* message)
{
    if (UNLIKELY(shouldDump)) {
        dataLogLn(message);
        AST::dumpAST(program);
        return true;
    }

    return false;
}

static inline bool dumpASTAfterParsingIfNeeded(ShaderModule& program)
{
    return dumpASTIfNeeded(dumpASTAfterParsing, program, "AST after parsing");
}

static inline bool dumpASTBetweenEachPassIfNeeded(ShaderModule& program, const char* message)
{
    return dumpASTIfNeeded(dumpASTBeforeEachPass, program, message);
}

static inline bool dumpASTAtEndIfNeeded(ShaderModule& program)
{
    return dumpASTIfNeeded(dumpASTAtEnd, program, "AST at end");
}

using PhaseTimes = Vector<std::pair<const char*, Seconds>>;

static inline void logPhaseTimes(PhaseTimes& phaseTimes)
{
    if (LIKELY(!dumpPhaseTimes))
        return;

    for (auto& entry : phaseTimes)
        dataLogLn(entry.first, ": ", entry.second.milliseconds(), " ms");
}

class PhaseTimer {
public:
    PhaseTimer(const char* phaseName, PhaseTimes& phaseTimes)
        : m_phaseTimes(phaseTimes)
    {
        if (dumpPhaseTimes) {
            m_phaseName = phaseName;
            m_start = MonotonicTime::now();
        }
    }

    ~PhaseTimer()
    {
        if (dumpPhaseTimes) {
            auto totalTime = MonotonicTime::now() - m_start;
            m_phaseTimes.append({ m_phaseName, totalTime });
        }
    }

private:
    const char* m_phaseName { nullptr };
    PhaseTimes& m_phaseTimes;
    MonotonicTime m_start;
};

} // namespace WGSL
