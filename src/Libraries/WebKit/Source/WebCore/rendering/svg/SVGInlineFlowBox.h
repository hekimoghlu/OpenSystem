/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#include "LegacyInlineFlowBox.h"
#include "RenderSVGInline.h"

namespace WebCore {

class RenderSVGInlineText;

class SVGInlineFlowBox final : public LegacyInlineFlowBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGInlineFlowBox);
public:
    SVGInlineFlowBox(RenderSVGInline& renderer)
        : LegacyInlineFlowBox(renderer)
        , m_logicalHeight(0)
    {
    }

    RenderSVGInline& renderer() { return static_cast<RenderSVGInline&>(LegacyInlineFlowBox::renderer()); }

    FloatRect calculateBoundaries() const;

    void setLogicalHeight(float h) { m_logicalHeight = h; }

private:
    bool isSVGInlineFlowBox() const override { return true; }
    float virtualLogicalHeight() const override { return m_logicalHeight; }

    float m_logicalHeight;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INLINE_BOX(SVGInlineFlowBox, isSVGInlineFlowBox())
