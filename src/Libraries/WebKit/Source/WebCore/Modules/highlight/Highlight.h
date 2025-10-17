/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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

#include "ExceptionOr.h"
#include "Position.h"
#include "Range.h"
#include "StaticRange.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class CSSStyleDeclaration;
class DOMSetAdapter;
class PropertySetCSSStyleDeclaration;

class HighlightRange : public RefCountedAndCanMakeWeakPtr<HighlightRange> {
public:
    static Ref<HighlightRange> create(Ref<AbstractRange>&& range)
    {
        return adoptRef(*new HighlightRange(WTFMove(range)));
    }

    AbstractRange& range() const { return m_range.get(); }
    const Position& startPosition() const { return m_startPosition; }
    void setStartPosition(Position&& startPosition) { m_startPosition = WTFMove(startPosition); }
    const Position& endPosition() const { return m_endPosition; }
    void setEndPosition(Position&& endPosition) { m_endPosition = WTFMove(endPosition); }

private:
    explicit HighlightRange(Ref<AbstractRange>&& range)
        : m_range(WTFMove(range))
    {
        if (auto liveRange = dynamicDowncast<Range>(m_range))
            liveRange->didAssociateWithHighlight();
    }

    Ref<AbstractRange> m_range;
    Position m_startPosition;
    Position m_endPosition;
};

class Highlight : public RefCounted<Highlight> {
public:
    WEBCORE_EXPORT static Ref<Highlight> create(FixedVector<std::reference_wrapper<AbstractRange>>&&);
    static void repaintRange(const AbstractRange&);
    void clearFromSetLike();
    bool addToSetLike(AbstractRange&);
    bool removeFromSetLike(const AbstractRange&);
    void initializeSetLike(DOMSetAdapter&);

    enum class Type : uint8_t { Highlight, SpellingError, GrammarError };
    Type type() const { return m_type; }
    void setType(Type type) { m_type = type; }

    int priority() const { return m_priority; }
    void setPriority(int);

    void repaint();
    const Vector<Ref<HighlightRange>>& highlightRanges() const { return m_highlightRanges; }

private:
    explicit Highlight(FixedVector<std::reference_wrapper<AbstractRange>>&&);

    Vector<Ref<HighlightRange>> m_highlightRanges;
    Type m_type { Type::Highlight };
    int m_priority { 0 };
};

} // namespace WebCore
