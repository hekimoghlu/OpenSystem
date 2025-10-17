/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#include "CompilerTimingScope.h"

#include "Options.h"
#include <wtf/DataLog.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace JSC {

namespace {

class CompilerTimingScopeState {
    WTF_MAKE_NONCOPYABLE(CompilerTimingScopeState);
    WTF_MAKE_TZONE_ALLOCATED(CompilerTimingScopeState);
public:
    CompilerTimingScopeState() { }
    
    Seconds addToTotal(const char* compilerName, const char* name, Seconds duration)
    {
        Locker locker { lock };

        for (auto& tuple : totals) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
            if (!strcmp(std::get<0>(tuple), compilerName) && !strcmp(std::get<1>(tuple), name)) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
                std::get<2>(tuple) += duration;
                std::get<3>(tuple) = std::max(std::get<3>(tuple), duration);
                return std::get<2>(tuple);
            }
        }

        totals.append({ compilerName, name, duration, duration });
        return duration;
    }

    void logTotals()
    {
        for (auto& tuple : totals) {
            dataLogLn(
                "total ms: ", FixedWidthDouble(std::get<2>(tuple).milliseconds(), 8, 3), " max ms: ", FixedWidthDouble(std::get<3>(tuple).milliseconds(), 7, 3), " [", std::get<0>(tuple), "] ", std::get<1>(tuple));
        }
    }
    
private:
    Vector<std::tuple<const char*, const char*, Seconds, Seconds>> totals;
    Lock lock;
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(CompilerTimingScopeState);

CompilerTimingScopeState& compilerTimingScopeState()
{
    static Atomic<CompilerTimingScopeState*> s_state;
    return ensurePointer(s_state, [] { return new CompilerTimingScopeState(); });
}

} // anonymous namespace

CompilerTimingScope::CompilerTimingScope(ASCIILiteral compilerName, ASCIILiteral name)
    : m_compilerName(compilerName)
    , m_name(name)
{
    if (UNLIKELY(Options::logPhaseTimes() || Options::reportTotalPhaseTimes()))
        m_before = MonotonicTime::now();
}

CompilerTimingScope::~CompilerTimingScope()
{
    if (UNLIKELY(Options::logPhaseTimes() || Options::reportTotalPhaseTimes())) {
        Seconds duration = MonotonicTime::now() - m_before;
        auto total = compilerTimingScopeState().addToTotal(m_compilerName, m_name, duration);
        if (Options::logPhaseTimes()) {
            dataLog(
                "[", m_compilerName, "] ", m_name, " took: ", duration.milliseconds(), " ms ",
                "(total: ", total.milliseconds(),
                " ms).\n");
        }
    }
}

void logTotalPhaseTimes()
{
    compilerTimingScopeState().logTotals();
}

} // namespace JSC
