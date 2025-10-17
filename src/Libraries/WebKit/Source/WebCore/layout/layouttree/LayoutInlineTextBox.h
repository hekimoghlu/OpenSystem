/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

#include "LayoutBox.h"
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

namespace Layout {

class InlineTextBox : public Box {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InlineTextBox);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InlineTextBox);
public:
    enum class ContentCharacteristic : uint8_t {
        CanUseSimplifiedContentMeasuring         = 1 << 0,
        CanUseSimpleFontCodepath                 = 1 << 1,
        ShouldUseSimpleGlyphOverflowCodePath     = 1 << 2,
        HasPositionDependentContentWidth         = 1 << 3,
        HasStrongDirectionalityContent           = 1 << 4
    };
    InlineTextBox(String, bool isCombined, OptionSet<ContentCharacteristic>, RenderStyle&&, std::unique_ptr<RenderStyle>&& firstLineStyle = nullptr);
    virtual ~InlineTextBox() = default;

    const String& content() const { return m_content; }
    bool isCombined() const { return m_isCombined; }
    // FIXME: This should not be a box's property.
    bool canUseSimplifiedContentMeasuring() const { return m_contentCharacteristicSet.contains(ContentCharacteristic::CanUseSimplifiedContentMeasuring); }
    bool canUseSimpleFontCodePath() const { return m_contentCharacteristicSet.contains(ContentCharacteristic::CanUseSimpleFontCodepath); }
    bool shouldUseSimpleGlyphOverflowCodePath() const { return m_contentCharacteristicSet.contains(ContentCharacteristic::ShouldUseSimpleGlyphOverflowCodePath); }
    bool hasPositionDependentContentWidth() const { return m_contentCharacteristicSet.contains(ContentCharacteristic::HasPositionDependentContentWidth); }
    bool hasStrongDirectionalityContent() const { return m_contentCharacteristicSet.contains(ContentCharacteristic::HasStrongDirectionalityContent); }

    void setContent(String newContent, OptionSet<ContentCharacteristic>);
    void setContentCharacteristic(OptionSet<ContentCharacteristic> contentCharacteristicSet) { m_contentCharacteristicSet = contentCharacteristicSet; }

private:
    String m_content;
    bool m_isCombined { false };
    OptionSet<ContentCharacteristic> m_contentCharacteristicSet;
};

inline void InlineTextBox::setContent(String newContent, OptionSet<ContentCharacteristic> contentCharacteristicSet)
{
    m_content = newContent;
    m_contentCharacteristicSet = contentCharacteristicSet;
}

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_BOX(InlineTextBox, isInlineTextBox())

