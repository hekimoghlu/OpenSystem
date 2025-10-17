/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

#include "RenderObject.h"

namespace WebCore {

struct TextBoxSelectableRange;

class HighlightRange;
class RenderMultiColumnSpannerPlaceholder;
class RenderText;

class RenderRange {
public:
    RenderRange() = default;
    RenderRange(RenderObject* start, RenderObject* end, unsigned startOffset, unsigned endOffset)
        : m_start(start)
        , m_end(end)
        , m_startOffset(startOffset)
        , m_endOffset(endOffset)
    {
    }

    RenderObject* start() const { return m_start.get(); }
    RenderObject* end() const { return m_end.get(); }
    unsigned startOffset() const { return m_startOffset; }
    unsigned endOffset() const { return m_endOffset; }

    friend bool operator==(const RenderRange&, const RenderRange&) = default;

private:
    SingleThreadWeakPtr<RenderObject> m_start;
    SingleThreadWeakPtr<RenderObject> m_end;
    unsigned m_startOffset { 0 };
    unsigned m_endOffset { 0 };
};
    
class RenderRangeIterator {
public:
    RenderRangeIterator(RenderObject* start);
    RenderObject* current() const;
    RenderObject* next();
    
private:
    void checkForSpanner();
    
    RenderObject* m_current { nullptr };
    Vector<RenderMultiColumnSpannerPlaceholder*> m_spannerStack;
};

class RenderHighlight {
public:
    enum IsSelectionTag { IsSelection };
    RenderHighlight() = default;
    RenderHighlight(IsSelectionTag)
        : m_isSelection(true)
    { }

    void setRenderRange(const RenderRange&);
    bool setRenderRange(const HighlightRange&); // Returns true if successful.
    const RenderRange& get() const { return m_renderRange; }

    RenderObject* start() const { return m_renderRange.start(); }
    RenderObject* end() const { return m_renderRange.end(); }

    unsigned startOffset() const { return m_renderRange.startOffset(); }
    unsigned endOffset() const { return m_renderRange.endOffset(); }

    RenderObject::HighlightState highlightStateForRenderer(const RenderObject&);
    RenderObject::HighlightState highlightStateForTextBox(const RenderText&, const TextBoxSelectableRange&);
    std::pair<unsigned, unsigned> rangeForTextBox(const RenderText&, const TextBoxSelectableRange&);

protected:
    RenderRange m_renderRange;
    const bool m_isSelection { false };
};

} // namespace WebCore
