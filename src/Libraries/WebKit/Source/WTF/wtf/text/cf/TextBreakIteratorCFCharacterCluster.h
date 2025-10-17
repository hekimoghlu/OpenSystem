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
#pragma once

#include <wtf/RetainPtr.h>
#include <wtf/spi/cf/CFStringSPI.h>
#include <wtf/text/StringView.h>
#include <wtf/text/cocoa/ContextualizedCFString.h>

namespace WTF {

class TextBreakIteratorCFCharacterCluster {
    WTF_MAKE_FAST_ALLOCATED;
public:
    enum class Mode {
        ComposedCharacter,
        BackwardDeletion
    };

    TextBreakIteratorCFCharacterCluster(StringView string, StringView priorContext, Mode mode)
    {
        setText(string, priorContext);

        switch (mode) {
        case Mode::ComposedCharacter:
            m_type = kCFStringComposedCharacterCluster;
            break;
        case Mode::BackwardDeletion:
            m_type = kCFStringBackwardDeletionCluster;
            break;
        }
    }

    TextBreakIteratorCFCharacterCluster() = delete;
    TextBreakIteratorCFCharacterCluster(const TextBreakIteratorCFCharacterCluster&) = delete;
    TextBreakIteratorCFCharacterCluster(TextBreakIteratorCFCharacterCluster&&) = default;
    TextBreakIteratorCFCharacterCluster& operator=(const TextBreakIteratorCFCharacterCluster&) = delete;
    TextBreakIteratorCFCharacterCluster& operator=(TextBreakIteratorCFCharacterCluster&&) = default;

    void setText(StringView string, StringView priorContext)
    {
        if (priorContext.isEmpty())
            m_string = string.createCFStringWithoutCopying();
        else
            m_string = createContextualizedCFString(string, priorContext);
        m_stringLength = string.length();
        m_priorContextLength = priorContext.length();
    }

    std::optional<unsigned> preceding(unsigned location) const
    {
        if (!location)
            return { };
        if (location > m_stringLength)
            return m_stringLength;
        auto range = CFStringGetRangeOfCharacterClusterAtIndex(m_string.get(), location - 1 + m_priorContextLength, m_type);
        return std::max(static_cast<unsigned long>(range.location), m_priorContextLength) - m_priorContextLength;
    }

    std::optional<unsigned> following(unsigned location) const
    {
        if (location >= m_stringLength)
            return { };
        auto range = CFStringGetRangeOfCharacterClusterAtIndex(m_string.get(), location + m_priorContextLength, m_type);
        return range.location + range.length - m_priorContextLength;
    }

    bool isBoundary(unsigned location) const
    {
        if (location == m_stringLength)
            return true;
        auto range = CFStringGetRangeOfCharacterClusterAtIndex(m_string.get(), location + m_priorContextLength, m_type);
        return static_cast<unsigned long>(range.location) == location + m_priorContextLength;
    }

private:
    RetainPtr<CFStringRef> m_string;
    CFStringCharacterClusterType m_type;
    unsigned long m_stringLength { 0 };
    unsigned long m_priorContextLength { 0 };
};

}
