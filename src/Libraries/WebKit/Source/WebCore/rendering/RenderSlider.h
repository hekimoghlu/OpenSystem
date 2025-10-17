/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#include "RenderFlexibleBox.h"

namespace WebCore {

class HTMLInputElement;
class MouseEvent;

class RenderSlider final : public RenderFlexibleBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSlider);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSlider);
public:
    static const int defaultTrackLength;

    RenderSlider(HTMLInputElement&, RenderStyle&&);
    virtual ~RenderSlider();

    HTMLInputElement& element() const;

    bool inDragMode() const;

    double valueRatio() const;

private:
    ASCIILiteral renderName() const override { return "RenderSlider"_s; }

    LayoutUnit baselinePosition(FontBaseline, bool firstLine, LineDirectionMode, LinePositionMode = PositionOnContainingLine) const override;
    void computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const override;
    void computePreferredLogicalWidths() override;

    bool isFlexibleBoxImpl() const override { return true; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSlider, isRenderSlider())
