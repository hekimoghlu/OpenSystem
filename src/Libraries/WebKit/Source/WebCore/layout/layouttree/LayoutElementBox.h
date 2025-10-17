/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class CachedImage;
class RenderElement;
class RenderStyle;

namespace Layout {

class ElementBox : public Box {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ElementBox);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ElementBox);
public:
    ElementBox(ElementAttributes&&, RenderStyle&&, std::unique_ptr<RenderStyle>&& firstLineStyle = nullptr, OptionSet<BaseTypeFlag> = { ElementBoxFlag });

    enum class ListMarkerAttribute : uint8_t {
        Image = 1 << 0,
        Outside = 1 << 1,
    };
    ElementBox(ElementAttributes&&, OptionSet<ListMarkerAttribute>, RenderStyle&&, std::unique_ptr<RenderStyle>&& firstLineStyle = nullptr);

    struct ReplacedAttributes {
        LayoutSize intrinsicSize;
        std::optional<LayoutUnit> intrinsicRatio { };
        CachedImage* cachedImage { };
    };
    ElementBox(ElementAttributes&&, ReplacedAttributes&&, RenderStyle&&, std::unique_ptr<RenderStyle>&& firstLineStyle = nullptr);

    ~ElementBox();

    const Box* firstChild() const { return m_firstChild.get(); }
    const Box* firstInFlowChild() const;
    const Box* firstInFlowOrFloatingChild() const;
    const Box* firstOutOfFlowChild() const;
    const Box* lastChild() const { return m_lastChild.get(); }
    const Box* lastInFlowChild() const;
    const Box* lastInFlowOrFloatingChild() const;
    const Box* lastOutOfFlowChild() const;

    // FIXME: This is currently needed for style updates.
    Box* firstChild() { return m_firstChild.get(); }

    bool hasChild() const { return firstChild(); }
    bool hasInFlowChild() const { return firstInFlowChild(); }
    bool hasInFlowOrFloatingChild() const { return firstInFlowOrFloatingChild(); }
    bool hasOutOfFlowChild() const;

    void appendChild(UniqueRef<Box>);
    void insertChild(UniqueRef<Box>, Box* beforeChild = nullptr);
    void destroyChildren();

    void setBaselineForIntegration(LayoutUnit baseline) { m_baselineForIntegration = baseline; }
    std::optional<LayoutUnit> baselineForIntegration() const { return m_baselineForIntegration; }

    bool hasIntrinsicWidth() const;
    bool hasIntrinsicHeight() const;
    bool hasIntrinsicRatio() const;
    LayoutUnit intrinsicWidth() const;
    LayoutUnit intrinsicHeight() const;
    LayoutUnit intrinsicRatio() const;
    bool hasAspectRatio() const;

    void setListMarkerAttributes(OptionSet<ListMarkerAttribute> listMarkerAttributes) { m_replacedData->listMarkerAttributes = listMarkerAttributes; }

    bool isListMarkerImage() const { return m_replacedData && m_replacedData->listMarkerAttributes.contains(ListMarkerAttribute::Image); }
    bool isListMarkerOutside() const { return m_replacedData && m_replacedData->listMarkerAttributes.contains(ListMarkerAttribute::Outside); }

    // FIXME: This doesn't belong.
    CachedImage* cachedImage() const { return m_replacedData ? m_replacedData->cachedImage : nullptr; }

    RenderElement* rendererForIntegration() const;

private:
    friend class Box;

    struct ReplacedData {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        OptionSet<ListMarkerAttribute> listMarkerAttributes;
        std::optional<LayoutSize> intrinsicSize;
        std::optional<LayoutUnit> intrinsicRatio;
        CachedImage* cachedImage { nullptr };
    };

    std::unique_ptr<Box> m_firstChild;
    CheckedPtr<Box> m_lastChild;

    std::unique_ptr<ReplacedData> m_replacedData;
    std::optional<LayoutUnit> m_baselineForIntegration;
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_BOX(ElementBox, isElementBox())

