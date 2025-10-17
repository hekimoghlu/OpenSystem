/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#include "MutationEvent.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MutationEvent);

MutationEvent::MutationEvent()
    : Event(EventInterfaceType::MutationEvent)
{
}

MutationEvent::MutationEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, Node* relatedNode, const String& prevValue, const String& newValue)
    : Event(EventInterfaceType::MutationEvent, type, canBubble, cancelable)
    , m_relatedNode(relatedNode)
    , m_prevValue(prevValue)
    , m_newValue(newValue)
{
}

void MutationEvent::initMutationEvent(const AtomString& type, bool canBubble, bool cancelable, Node* relatedNode, const String& prevValue, const String& newValue, const String& attrName, unsigned short attrChange)
{
    if (isBeingDispatched())
        return;

    initEvent(type, canBubble, cancelable);

    m_relatedNode = relatedNode;
    m_prevValue = prevValue;
    m_newValue = newValue;
    m_attrName = attrName;
    m_attrChange = attrChange;
}

} // namespace WebCore
