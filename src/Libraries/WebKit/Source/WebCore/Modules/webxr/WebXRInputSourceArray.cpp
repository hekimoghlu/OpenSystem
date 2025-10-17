/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "WebXRInputSourceArray.h"

#if ENABLE(WEBXR)
#include "EventNames.h"
#include "WebXRInputSource.h"
#include "WebXRSession.h"
#include "XRInputSourceEvent.h"
#include "XRInputSourcesChangeEvent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRInputSourceArray);

WebXRInputSourceArray::WebXRInputSourceArray(WebXRSession& session)
    : m_session(session)
{
}

WebXRInputSourceArray::~WebXRInputSourceArray() = default;

void WebXRInputSourceArray::ref()
{
    m_session.ref();
}

void WebXRInputSourceArray::deref()
{
    m_session.deref();
}

unsigned WebXRInputSourceArray::length() const
{
    return m_inputSources.size();
}

WebXRInputSource* WebXRInputSourceArray::item(unsigned index) const
{
    return index >= m_inputSources.size() ? nullptr: m_inputSources[index].ptr();
}

void WebXRInputSourceArray::clear()
{
    m_inputSources.clear();
}

// https://immersive-web.github.io/webxr/#list-of-active-xr-input-sources
void WebXRInputSourceArray::update(double timestamp, const InputSourceList& inputSources)
{
    Vector<Ref<WebXRInputSource>> added;
    Vector<Ref<WebXRInputSource>> removed;
    Vector<Ref<WebXRInputSource>> removedWithInputEvents;
    Vector<Ref<XRInputSourceEvent>> inputEvents;

    handleRemovedInputSources(inputSources, removed, removedWithInputEvents, inputEvents);
    handleAddedOrUpdatedInputSources(timestamp, inputSources, added, removed, removedWithInputEvents, inputEvents);

    if (!added.isEmpty() || !removed.isEmpty()) {
        // A user agent MUST dispatch an inputsourceschange event on an XRSession when the sessionâ€™s list of active XR input sources has changed.
        XRInputSourcesChangeEvent::Init init;
        init.session = &m_session;
        init.added = WTFMove(added);
        init.removed = WTFMove(removed);
        
        auto event = XRInputSourcesChangeEvent::create(eventNames().inputsourceschangeEvent, init);
        ActiveDOMObject::queueTaskToDispatchEvent(m_session, TaskSource::WebXR, WTFMove(event));
    }

    if (!inputEvents.isEmpty()) {
        // When the user agent has to fire an input source event with name name, XRFrame frame, and XRInputSource source it MUST run the following steps:
        // 1. Create an XRInputSourceEvent event with type name, frame frame, and inputSource source.
        // 2. Set frameâ€™s active boolean to true.
        // 3. Apply frame updates for frame.
        // 4. Dispatch event on frameâ€™s session
        // 5. Set frameâ€™s active boolean to false.

        for (auto& event : inputEvents) {
            ActiveDOMObject::queueTaskKeepingObjectAlive(m_session, TaskSource::WebXR, [session = Ref { m_session }, event = WTFMove(event)]() {
                event->setFrameActive(true);
                session->dispatchEvent(event.copyRef());
                event->setFrameActive(false);
            });
        }
    }

    // If any input sources being removed need to fire any input source events, we need to
    // make sure the inputsourceschange event for the removal happen after the input source events.
    if (!removedWithInputEvents.isEmpty()) {
        // A user agent MUST dispatch an inputsourceschange event on an XRSession when the sessionâ€™s list of active XR input sources has changed.
        XRInputSourcesChangeEvent::Init init;
        init.session = &m_session;
        init.removed = WTFMove(removedWithInputEvents);

        auto event = XRInputSourcesChangeEvent::create(eventNames().inputsourceschangeEvent, init);
        ActiveDOMObject::queueTaskToDispatchEvent(m_session, TaskSource::WebXR, WTFMove(event));
    }
}

// https://immersive-web.github.io/webxr/#list-of-active-xr-input-sources
void WebXRInputSourceArray::handleRemovedInputSources(const InputSourceList& inputSources, Vector<Ref<WebXRInputSource>>& removed, Vector<Ref<WebXRInputSource>>& removedWithInputEvents, Vector<Ref<XRInputSourceEvent>>& inputEvents)
{
    // When any previously added XR input sources are no longer available for XRSession session, the user agent MUST run the following steps:
    // 1. If session's promise resolved flag is not set, abort these steps.
    // 2. Let removed be a new list.
    // 3. For each XR input source that is no longer available:
    //  3.1 Let inputSource be the XRInputSource in session's list of active XR input sources associated with the XR input source.
    //  3.2 Add inputSource to removed.
    m_inputSources.removeAllMatching([&inputSources, &removed, &removedWithInputEvents, &inputEvents](auto& source) {
        if (!WTF::anyOf(inputSources, [&source](auto& item) { return item.handle == source->handle(); })) {
            Vector<Ref<XRInputSourceEvent>> sourceInputEvents;
            source->disconnect();
            source->pollEvents(sourceInputEvents);
            if (sourceInputEvents.isEmpty())
                removed.append(source);
            else
                removedWithInputEvents.append(source);
            inputEvents.appendVector(sourceInputEvents);
            return true;
        }
        return false;
    });
}

// https://immersive-web.github.io/webxr/#list-of-active-xr-input-sources
void WebXRInputSourceArray::handleAddedOrUpdatedInputSources(double timestamp, const InputSourceList& inputSources, Vector<Ref<WebXRInputSource>>& added, Vector<Ref<WebXRInputSource>>& removed, Vector<Ref<WebXRInputSource>>& removedWithInputEvents, Vector<Ref<XRInputSourceEvent>>& inputEvents)
{
    RefPtr document = downcast<Document>(m_session.scriptExecutionContext());
    if (!document)
        return;

    for (auto& inputSource : inputSources) {
        auto index = m_inputSources.findIf([&inputSource](auto& item) { return item->handle() == inputSource.handle; });
        if (index == notFound) {
            // When new XR input sources become available for XRSession session, the user agent MUST run the following steps:
            // 1. If session's promise resolved flag is not set, abort these steps.
            // 2. Let added be a new list.
            // 3. For each new XR input source:
            //   3.1 Let inputSource be a new XRInputSource in the relevant realm of this XRSession.
            //   3.2 Add inputSource to added.

            auto input = WebXRInputSource::create(*document, m_session, timestamp, inputSource);
            added.append(input);
            input->pollEvents(inputEvents);
            m_inputSources.append(WTFMove(input));
            continue;
        }

        // When the handedness, targetRayMode, profiles, or presence of a gripSpace for any XR input sources change for XRSession session, the user agent MUST run the following steps
        // 1. If sessionâ€™s promise resolved flag is not set, abort these steps.
        // 2. Let added be a new list.
        // 3. Let removed be a new list.
        // 4. For each changed XR input source:
        //  4.1 Let oldInputSource be the XRInputSource in session's list of active XR input sources previously associated with the XR input source.
        //  4.1 Let newInputSource be a new XRInputSource in the relevant realm of session.
        //  4.1 Add oldInputSource to removed.
        //  4.1 Add newInputSource to added.
        Ref input = m_inputSources[index];

        if (input->requiresInputSourceChange(inputSource)) {
            Vector<Ref<XRInputSourceEvent>> sourceInputEvents;
            input->disconnect();
            input->pollEvents(sourceInputEvents);
            if (sourceInputEvents.isEmpty())
                removed.append(input);
            else
                removedWithInputEvents.append(input);
            inputEvents.appendVector(sourceInputEvents);
            m_inputSources.remove(index);

            auto newInputSource = WebXRInputSource::create(*document, m_session, timestamp, inputSource);
            added.append(newInputSource);
            newInputSource->pollEvents(inputEvents);
            m_inputSources.append(WTFMove(newInputSource));
        } else {
            input->update(timestamp, inputSource);
            input->pollEvents(inputEvents);
        }
    }
}

} // namespace WebCore

#endif // ENABLE(WEBXR)

