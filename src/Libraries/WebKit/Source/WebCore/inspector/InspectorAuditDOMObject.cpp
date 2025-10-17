/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#include "InspectorAuditDOMObject.h"

#include "Document.h"
#include "Node.h"
#include "UserGestureEmulationScope.h"
#include "VoidCallback.h"
#include <wtf/text/AtomString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using namespace Inspector;

#define ERROR_IF_NO_ACTIVE_AUDIT() \
    if (!m_auditAgent.hasActiveAudit()) \
        return Exception { ExceptionCode::NotAllowedError, "Cannot be called outside of a Web Inspector Audit"_s };

InspectorAuditDOMObject::InspectorAuditDOMObject(PageAuditAgent& auditAgent)
    : m_auditAgent(auditAgent)
{
}

ExceptionOr<bool> InspectorAuditDOMObject::hasEventListeners(Node& node, const String& type)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    if (EventTargetData* eventTargetData = node.eventTargetData()) {
        Vector<AtomString> eventTypes;
        if (type.isNull())
            eventTypes = eventTargetData->eventListenerMap.eventTypes();
        else
            eventTypes.append(type);

        for (AtomString& type : eventTypes) {
            for (const RefPtr<RegisteredEventListener>& listener : node.eventListeners(type)) {
                if (listener->callback().type() == EventListener::JSEventListenerType)
                    return true;
            }
        }
    }

    return false;
}

ExceptionOr<void> InspectorAuditDOMObject::simulateUserInteraction(Document& document, Ref<VoidCallback>&& callback)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    UserGestureEmulationScope userGestureScope(m_auditAgent.inspectedPage(), true, &document);
    callback->handleEvent();

    return { };
}

} // namespace WebCore
