/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "ICStats.h"

#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ICStats);

bool ICEvent::operator<(const ICEvent& other) const
{
    if (m_classInfo != other.m_classInfo) {
        if (!m_classInfo)
            return true;
        if (!other.m_classInfo)
            return false;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        return strcmp(m_classInfo->className, other.m_classInfo->className) < 0;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }
    
    if (m_propertyName != other.m_propertyName)
        return codePointCompare(m_propertyName.string(), other.m_propertyName.string()) < 0;

    if (m_kind != other.m_kind)
        return m_kind < other.m_kind;

    return m_propertyLocation < other.m_propertyLocation;
}

void ICEvent::dump(PrintStream& out) const
{
    out.print(m_kind, "(", m_classInfo ? m_classInfo->className : "<null>", ", ", m_propertyName, ")");
    if (m_propertyLocation != Unknown)
        out.print(m_propertyLocation == BaseObject ? " self" : " proto lookup");
}

void ICEvent::log() const
{
    ICStats::singleton().add(*this);
}

Atomic<ICStats*> ICStats::s_instance;

ICStats::ICStats()
{
    m_thread = Thread::create(
        "JSC ICStats"_s,
        [this] () {
            Locker locker { m_lock };
            for (;;) {
                m_condition.waitFor(
                    m_lock, Seconds(1), [this] () -> bool { return m_shouldStop; });
                if (m_shouldStop)
                    break;
                
                dataLog("ICStats:\n");
                {
                    Locker spectrumLocker { m_spectrum.getLock() };
                    auto list = m_spectrum.buildList(spectrumLocker);
                    for (unsigned i = list.size(); i--;)
                        dataLog("    ", *list[i].key, ": ", list[i].count, "\n");
                }
            }
        });
}

ICStats::~ICStats()
{
    {
        Locker locker { m_lock };
        m_shouldStop = true;
        m_condition.notifyAll();
    }
    
    m_thread->waitForCompletion();
}

void ICStats::add(const ICEvent& event)
{
    m_spectrum.add(event);
}

ICStats& ICStats::singleton()
{
    for (;;) {
        ICStats* result = s_instance.load();
        if (result)
            return *result;
        
        ICStats* newStats = new ICStats();
        if (s_instance.compareExchangeWeak(nullptr, newStats))
            return *newStats;
        
        delete newStats;
    }
}

} // namespace JSC

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, ICEvent::Kind kind)
{
    switch (kind) {
#define ICEVENT_KIND_DUMP(name) case ICEvent::name: out.print(#name); return;
        FOR_EACH_ICEVENT_KIND(ICEVENT_KIND_DUMP);
#undef ICEVENT_KIND_DUMP
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF


