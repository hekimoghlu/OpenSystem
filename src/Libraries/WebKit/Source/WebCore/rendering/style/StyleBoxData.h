/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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

#include "Length.h"
#include "RenderStyleConstants.h"
#include <wtf/RefCounted.h>
#include <wtf/Ref.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleBoxData);
class StyleBoxData : public RefCounted<StyleBoxData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleBoxData);
public:
    static Ref<StyleBoxData> create() { return adoptRef(*new StyleBoxData); }
    Ref<StyleBoxData> copy() const;

    bool operator==(const StyleBoxData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleBoxData&) const;
#endif

    const Length& width() const { return m_width; }
    const Length& height() const { return m_height; }
    
    const Length& minWidth() const { return m_minWidth; }
    const Length& minHeight() const { return m_minHeight; }
    
    const Length& maxWidth() const { return m_maxWidth; }
    const Length& maxHeight() const { return m_maxHeight; }
    
    const Length& verticalAlignLength() const { return m_verticalAlignLength; }
    
    int specifiedZIndex() const { return m_specifiedZIndex; }
    bool hasAutoSpecifiedZIndex() const { return m_hasAutoSpecifiedZIndex; }

    int usedZIndex() const { return m_usedZIndex; }
    bool hasAutoUsedZIndex() const { return m_hasAutoUsedZIndex; }

    BoxSizing boxSizing() const { return static_cast<BoxSizing>(m_boxSizing); }
    BoxDecorationBreak boxDecorationBreak() const { return static_cast<BoxDecorationBreak>(m_boxDecorationBreak); }
    VerticalAlign verticalAlign() const { return static_cast<VerticalAlign>(m_verticalAlign); }

private:
    friend class RenderStyle;

    StyleBoxData();
    StyleBoxData(const StyleBoxData&);

    Length m_width;
    Length m_height;

    Length m_minWidth;
    Length m_maxWidth;

    Length m_minHeight;
    Length m_maxHeight;

    Length m_verticalAlignLength;

    int m_specifiedZIndex;
    int m_usedZIndex;
    unsigned m_hasAutoSpecifiedZIndex : 1;
    unsigned m_hasAutoUsedZIndex : 1;
    unsigned m_boxSizing : 1; // BoxSizing
    unsigned m_boxDecorationBreak : 1; // BoxDecorationBreak
    unsigned m_verticalAlign : 4; // VerticalAlign
};

} // namespace WebCore
