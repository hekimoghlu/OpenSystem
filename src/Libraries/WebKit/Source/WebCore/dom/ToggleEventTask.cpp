/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
#include "ToggleEventTask.h"

#include "EventNames.h"
#include "ToggleEvent.h"

namespace WebCore {

Ref<ToggleEventTask> ToggleEventTask::create(Element& element)
{
    return adoptRef(*new ToggleEventTask(element));
}

void ToggleEventTask::queue(ToggleState oldState, ToggleState newState)
{
    if (m_data)
        oldState = m_data->oldState;

    RefPtr element = m_element.get();
    if (!element)
        return;

    m_data = { oldState, newState };
    element->queueTaskKeepingThisNodeAlive(TaskSource::DOMManipulation, [task = Ref { *this }, element, newState] {
        if (!task->m_data || task->m_data->newState != newState)
            return;

        auto stringForState = [](ToggleState state) {
            return state == ToggleState::Closed ? "closed"_s : "open"_s;
        };

        auto data = *std::exchange(task->m_data, std::nullopt);
        element->dispatchEvent(ToggleEvent::create(eventNames().toggleEvent, { EventInit { }, stringForState(data.oldState), stringForState(data.newState) }, Event::IsCancelable::No));
    });
}

} // namespace WebCore
