/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#include "JITSafepoint.h"

#if ENABLE(JIT)

#include "JITPlan.h"
#include "JITScannable.h"
#include "JITWorklistThread.h"
#include "SlotVisitor.h"

namespace JSC {

Safepoint::Result::~Result()
{
    RELEASE_ASSERT(m_wasChecked);
}

bool Safepoint::Result::didGetCancelled()
{
    m_wasChecked = true;
    return m_didGetCancelled;
}

Safepoint::Safepoint(JITPlan& plan, Result& result)
    : m_vm(plan.vm())
    , m_plan(plan)
    , m_didCallBegin(false)
    , m_result(result)
{
    RELEASE_ASSERT(result.m_wasChecked);
    result.m_wasChecked = false;
    result.m_didGetCancelled = false;
}

Safepoint::~Safepoint()
{
    RELEASE_ASSERT(m_didCallBegin);
    if (JITWorklistThread* thread = m_plan.thread()) {
        RELEASE_ASSERT(thread->m_safepoint == this);
        thread->m_rightToRun.lock();
        thread->m_safepoint = nullptr;
    }
}

void Safepoint::add(Scannable* scannable)
{
    RELEASE_ASSERT(!m_didCallBegin);
    m_scannables.append(scannable);
}

void Safepoint::begin(bool keepDependenciesLive) WTF_IGNORES_THREAD_SAFETY_ANALYSIS
{
    RELEASE_ASSERT(!m_didCallBegin);
    m_didCallBegin = true;
    m_keepDependenciesLive = keepDependenciesLive;
    if (JITWorklistThread* data = m_plan.thread()) {
        RELEASE_ASSERT(!data->m_safepoint);
        data->m_safepoint = this;
        data->m_rightToRun.unlockFairly();
    }
}

template<typename Visitor>
void Safepoint::checkLivenessAndVisitChildren(Visitor& visitor)
{
    RELEASE_ASSERT(m_didCallBegin);

    if (m_result.m_didGetCancelled)
        return; // We were cancelled during a previous GC!
    
    if (!isKnownToBeLiveDuringGC(visitor))
        return;
    
    for (unsigned i = m_scannables.size(); i--;)
        m_scannables[i]->visitChildren(visitor);
}

template void Safepoint::checkLivenessAndVisitChildren(AbstractSlotVisitor&);
template void Safepoint::checkLivenessAndVisitChildren(SlotVisitor&);

template<typename Visitor>
bool Safepoint::isKnownToBeLiveDuringGC(Visitor& visitor)
{
    RELEASE_ASSERT(m_didCallBegin);
    
    if (m_result.m_didGetCancelled)
        return true; // We were cancelled during a previous GC, so let's not mess with it this time around - pretend it's live and move on.

    return m_plan.isKnownToBeLiveDuringGC(visitor);
}

template bool Safepoint::isKnownToBeLiveDuringGC(AbstractSlotVisitor&);
template bool Safepoint::isKnownToBeLiveDuringGC(SlotVisitor&);

bool Safepoint::isKnownToBeLiveAfterGC()
{
    RELEASE_ASSERT(m_didCallBegin);
    
    if (m_result.m_didGetCancelled)
        return true; // We were cancelled during a previous GC, so let's not mess with it this time around - pretend it's live and move on.
    
    return m_plan.isKnownToBeLiveAfterGC();
}

void Safepoint::cancel()
{
    RELEASE_ASSERT(m_didCallBegin);
    RELEASE_ASSERT(!m_result.m_didGetCancelled); // We cannot get cancelled twice because subsequent GCs will think that we're alive and they will not do anything to us.
    
    RELEASE_ASSERT(m_plan.stage() == JITPlanStage::Canceled);
    m_result.m_didGetCancelled = true;
    m_vm = nullptr;
}

bool Safepoint::keepDependenciesLive() const
{
    return m_keepDependenciesLive;
}

VM* Safepoint::vm() const
{
    return m_vm;
}

} // namespace JSC

#endif // ENABLE(JIT)

