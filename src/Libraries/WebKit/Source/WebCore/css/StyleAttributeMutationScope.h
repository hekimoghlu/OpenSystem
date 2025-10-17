/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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

#include "CustomElementReactionQueue.h"
#include "Element.h"
#include "HTMLNames.h"
#include "MutationObserverInterestGroup.h"
#include "MutationRecord.h"

namespace WebCore {

class StyleAttributeMutationScope {
    WTF_MAKE_NONCOPYABLE(StyleAttributeMutationScope);
public:
    StyleAttributeMutationScope(Element* element)
        : m_element(element)
    {
        ++s_scopeCount;

        if (s_scopeCount != 1) {
            ASSERT(s_currentScope->m_element == element);
            return;
        }

        ASSERT(!s_currentScope);
        s_currentScope = this;

        if (!m_element)
            return;

        bool shouldReadOldValue = false;

        m_mutationRecipients = MutationObserverInterestGroup::createForAttributesMutation(*m_element, HTMLNames::styleAttr);
        if (m_mutationRecipients && m_mutationRecipients->isOldValueRequested())
            shouldReadOldValue = true;

        if (UNLIKELY(m_element->isDefinedCustomElement())) {
            auto* reactionQueue = m_element->reactionQueue();
            if (reactionQueue && reactionQueue->observesStyleAttribute()) {
                m_isCustomElement = true;
                shouldReadOldValue = true;
            }
        }

        if (shouldReadOldValue)
            m_oldValue = m_element->getAttribute(HTMLNames::styleAttr);
    }

    ~StyleAttributeMutationScope()
    {
        --s_scopeCount;
        if (s_scopeCount)
            return;
        ASSERT(s_currentScope == this);
        s_currentScope = nullptr;

        if (!m_shouldDeliver || !m_element)
            return;

        if (m_mutationRecipients) {
            auto mutation = MutationRecord::createAttributes(*m_element, HTMLNames::styleAttr, m_oldValue);
            m_mutationRecipients->enqueueMutationRecord(WTFMove(mutation));
        }

        if (m_isCustomElement) {
            auto& newValue = m_element->getAttribute(HTMLNames::styleAttr);
            CustomElementReactionQueue::enqueueAttributeChangedCallbackIfNeeded(*m_element, HTMLNames::styleAttr, m_oldValue, newValue);
        }
    }

    void enqueueMutationRecord()
    {
        m_shouldDeliver = true;
    }

private:
    static unsigned s_scopeCount;
    static StyleAttributeMutationScope* s_currentScope;

    std::unique_ptr<MutationObserverInterestGroup> m_mutationRecipients;
    AtomString m_oldValue;
    RefPtr<Element> m_element;
    bool m_isCustomElement { false };
    bool m_shouldDeliver { false };
};

} // namespace WebCore
