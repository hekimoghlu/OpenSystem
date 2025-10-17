/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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

#include "LegacyInlineTextBox.h"

namespace WebCore {

class RenderSVGInlineText;
class SVGRootInlineBox;
struct SVGTextFragment;

class SVGInlineTextBox final : public LegacyInlineTextBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGInlineTextBox);
public:
    explicit SVGInlineTextBox(RenderSVGInlineText&);

    inline RenderSVGInlineText& renderer() const;

    float virtualLogicalHeight() const override { return m_logicalHeight; }
    void setLogicalHeight(float height) { m_logicalHeight = height; }

    bool mapStartEndPositionsIntoFragmentCoordinates(const SVGTextFragment&, unsigned& startPosition, unsigned& endPosition) const;

    FloatRect calculateBoundaries() const;

    const Vector<SVGTextFragment>& textFragments() const { return m_textFragments; }
    void setTextFragments(Vector<SVGTextFragment>&&);

    void dirtyOwnLineBoxes() override;
    void dirtyLineBoxes() override;

    inline SVGInlineTextBox* nextTextBox() const;
    
private:
    bool isSVGInlineTextBox() const override { return true; }

private:
    float m_logicalHeight { 0 };

    Vector<SVGTextFragment> m_textFragments;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INLINE_BOX(SVGInlineTextBox, isSVGInlineTextBox())
