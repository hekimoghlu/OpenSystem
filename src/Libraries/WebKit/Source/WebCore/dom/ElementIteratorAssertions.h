/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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

#include "Document.h"
#include "ScriptDisallowedScope.h"

namespace WebCore {

class ElementIteratorAssertions {
public:
    ElementIteratorAssertions(const Node* first = nullptr);
    bool domTreeHasMutated() const;
    void dropEventDispatchAssertion();
    void clear();

private:
    WeakPtr<const Document, WeakPtrImplWithEventTargetData> m_document;
    uint64_t m_initialDOMTreeVersion;
    std::optional<ScriptDisallowedScope> m_eventDispatchAssertion;
};

// FIXME: No real point in doing these as inlines; they are for debugging and we usually turn off inlining in debug builds.

inline ElementIteratorAssertions::ElementIteratorAssertions(const Node* first)
    : m_document(first ? &first->document() : nullptr)
    , m_initialDOMTreeVersion(first ? m_document->domTreeVersion() : 0)
{
    if (first)
        m_eventDispatchAssertion = ScriptDisallowedScope();
}

inline bool ElementIteratorAssertions::domTreeHasMutated() const
{
    return m_document && m_document->domTreeVersion() != m_initialDOMTreeVersion;
}

inline void ElementIteratorAssertions::dropEventDispatchAssertion()
{
    m_eventDispatchAssertion = std::nullopt;
}

inline void ElementIteratorAssertions::clear()
{
    m_document = nullptr;
    m_initialDOMTreeVersion = 0;
    m_eventDispatchAssertion = std::nullopt;
}

} // namespace WebCore
