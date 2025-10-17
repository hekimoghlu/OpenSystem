/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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

#if ENABLE(JIT)

#include "JITPlan.h"
#include <wtf/AutomaticThread.h>

namespace JSC {

class JITWorklist;
class Safepoint;

class JITWorklistThread final : public AutomaticThread {
    class WorkScope;

    friend class Safepoint;
    friend class WorkScope;
    friend class JITWorklist;

    enum class State : uint8_t {
        NotCompiling,
        Compiling,
    };

public:
    JITWorklistThread(const AbstractLocker&, JITWorklist&);

    ASCIILiteral name() const final;

    State state() const { return m_state; }
    const Safepoint* safepoint() const { return m_safepoint; }

private:
    PollResult poll(const AbstractLocker&) final;
    WorkResult work() final;

    void threadDidStart() final;

    void threadIsStopping(const AbstractLocker&) final;

    Lock m_rightToRun;
    State m_state { State::NotCompiling };
    JITWorklist& m_worklist;
    RefPtr<JITPlan> m_plan { nullptr };
    Safepoint* m_safepoint { nullptr };
};

} // namespace JSC

#endif // ENABLE(JIT)
