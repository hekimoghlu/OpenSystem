/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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

#include "Event.h"
#include "Node.h"

namespace WebCore {

class MutationEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MutationEvent);
public:
    enum {
        MODIFICATION = 1,
        ADDITION = 2,
        REMOVAL = 3
    };

    static Ref<MutationEvent> create(const AtomString& type, CanBubble canBubble, Node* relatedNode = nullptr, const String& prevValue = String(), const String& newValue = String())
    {
        return adoptRef(*new MutationEvent(type, canBubble, IsCancelable::No, relatedNode, prevValue, newValue));
    }

    static Ref<MutationEvent> createForBindings()
    {
        return adoptRef(*new MutationEvent);
    }

    WEBCORE_EXPORT void initMutationEvent(const AtomString& type, bool canBubble, bool cancelable, Node* relatedNode, const String& prevValue, const String& newValue, const String& attrName, unsigned short attrChange);

    Node* relatedNode() const { return m_relatedNode.get(); }
    String prevValue() const { return m_prevValue; }
    String newValue() const { return m_newValue; }
    String attrName() const { return m_attrName; }
    unsigned short attrChange() const { return m_attrChange; }

private:
    MutationEvent();
    MutationEvent(const AtomString& type, CanBubble, IsCancelable, Node* relatedNode, const String& prevValue, const String& newValue);

    RefPtr<Node> m_relatedNode;
    String m_prevValue;
    String m_newValue;
    String m_attrName;
    unsigned short m_attrChange { 0 };
};

} // namespace WebCore
