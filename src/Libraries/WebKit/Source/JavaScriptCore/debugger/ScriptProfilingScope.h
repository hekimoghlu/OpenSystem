/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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

#include "Debugger.h"

namespace JSC {

class ScriptProfilingScope {
public:
    ScriptProfilingScope(JSGlobalObject* globalObject, ProfilingReason reason)
        : m_globalObject(globalObject)
        , m_reason(reason)
    {
        if (shouldStartProfile())
            m_startTime = m_globalObject->debugger()->willEvaluateScript();
    }

    ~ScriptProfilingScope()
    {
        if (shouldEndProfile())
            m_globalObject->debugger()->didEvaluateScript(m_startTime.value(), m_reason);
    }

private:
    bool shouldStartProfile() const
    {
        if (!m_globalObject)
            return false;

        if (!m_globalObject->hasDebugger())
            return false;

        if (!m_globalObject->debugger()->hasProfilingClient())
            return false;

        if (m_globalObject->debugger()->isAlreadyProfiling())
            return false;

        return true;
    }

    bool shouldEndProfile() const
    {
        // Did not start a profile.
        if (!m_startTime)
            return false;

        // Debugger may have been removed.
        if (!m_globalObject->hasDebugger())
            return false;

        // Profiling Client may have been removed.
        if (!m_globalObject->debugger()->hasProfilingClient())
            return false;

        return true;
    }

    JSGlobalObject* const m_globalObject { nullptr };
    std::optional<Seconds> m_startTime;
    const ProfilingReason m_reason;
};

} // namespace JSC
