/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#include "SourceBufferList.h"

#if ENABLE(MEDIA_SOURCE)

#include "ContextDestructionObserverInlines.h"
#include "Event.h"
#include "EventNames.h"
#include "SourceBuffer.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SourceBufferList);

Ref<SourceBufferList> SourceBufferList::create(ScriptExecutionContext* context)
{
    auto result = adoptRef(*new SourceBufferList(context));
    result->suspendIfNeeded();
    return result;
}

SourceBufferList::SourceBufferList(ScriptExecutionContext* context)
    : ActiveDOMObject(context)
{
}

SourceBufferList::~SourceBufferList()
{
    ASSERT(m_list.isEmpty());
}

void SourceBufferList::add(Ref<SourceBuffer>&& buffer)
{
    m_list.append(WTFMove(buffer));
    scheduleEvent(eventNames().addsourcebufferEvent);
}

bool SourceBufferList::contains(SourceBuffer& buffer) const
{
    return m_list.find(Ref { buffer } ) != notFound;
}

RefPtr<SourceBuffer> SourceBufferList::item(unsigned index) const
{
    if (index < m_list.size())
        return m_list[index].copyRef();
    return { };
}

void SourceBufferList::remove(SourceBuffer& buffer)
{
    size_t index = m_list.find(Ref { buffer });
    if (index == notFound)
        return;
    m_list.remove(index);
    scheduleEvent(eventNames().removesourcebufferEvent);
}

void SourceBufferList::clear()
{
    m_list.clear();
    scheduleEvent(eventNames().removesourcebufferEvent);
}

void SourceBufferList::replaceWith(Vector<Ref<SourceBuffer>>&& other)
{
    int changeInSize = other.size() - m_list.size();
    int addedEntries = 0;
    for (auto& sourceBuffer : other) {
        if (!m_list.contains(sourceBuffer))
            ++addedEntries;
    }
    int removedEntries = addedEntries - changeInSize;

    m_list = WTFMove(other);

    if (addedEntries)
        scheduleEvent(eventNames().addsourcebufferEvent);
    if (removedEntries)
        scheduleEvent(eventNames().removesourcebufferEvent);
}

void SourceBufferList::scheduleEvent(const AtomString& eventName)
{
    queueTaskToDispatchEvent(*this, TaskSource::MediaElement, Event::create(eventName, Event::CanBubble::No, Event::IsCancelable::No));
}

WebCoreOpaqueRoot root(SourceBufferList* list)
{
    return WebCoreOpaqueRoot { list };
}

} // namespace WebCore

#endif
