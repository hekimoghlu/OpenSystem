/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#include "config.h"
#include "SVGInlineTextBox.h"

#include "FloatConversion.h"
#include "GraphicsContext.h"
#include "HitTestResult.h"
#include "LegacyInlineFlowBox.h"
#include "LegacyRenderSVGResourceSolidColor.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "PointerEventsHitRules.h"
#include "RenderBlock.h"
#include "RenderInline.h"
#include "RenderSVGText.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include "SVGInlineTextBoxInlines.h"
#include "SVGRenderStyle.h"
#include "SVGRenderingContext.h"
#include "SVGResourcesCache.h"
#include "SVGRootInlineBox.h"
#include "SVGTextFragment.h"
#include "TextBoxSelectableRange.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGInlineTextBox);

SVGInlineTextBox::SVGInlineTextBox(RenderSVGInlineText& renderer)
    : LegacyInlineTextBox(renderer)
{
}

void SVGInlineTextBox::dirtyOwnLineBoxes()
{
    LegacyInlineTextBox::dirtyLineBoxes();

    m_textFragments = { };
}

void SVGInlineTextBox::dirtyLineBoxes()
{
    dirtyOwnLineBoxes();

    // And clear any following text fragments as the text on which they
    // depend may now no longer exist, or glyph positions may be wrong
    for (auto* nextBox = nextTextBox(); nextBox; nextBox = nextBox->nextTextBox())
        nextBox->dirtyOwnLineBoxes();
}

bool SVGInlineTextBox::mapStartEndPositionsIntoFragmentCoordinates(const SVGTextFragment& fragment, unsigned& startPosition, unsigned& endPosition) const
{
    unsigned startFragment = fragment.characterOffset - start();
    unsigned endFragment = startFragment + fragment.length;

    // Find intersection between the intervals: [startFragment..endFragment) and [startPosition..endPosition)
    startPosition = std::max(startFragment, startPosition);
    endPosition = std::min(endFragment, endPosition);

    if (startPosition >= endPosition)
        return false;

    startPosition -= startFragment;
    endPosition -= startFragment;

    return true;
}

FloatRect SVGInlineTextBox::calculateBoundaries() const
{
    FloatRect textRect;

    float scalingFactor = renderer().scalingFactor();
    ASSERT(scalingFactor);

    float baseline = renderer().scaledFont().metricsOfPrimaryFont().ascent() / scalingFactor;

    AffineTransform fragmentTransform;
    unsigned textFragmentsSize = m_textFragments.size();
    for (unsigned i = 0; i < textFragmentsSize; ++i) {
        const SVGTextFragment& fragment = m_textFragments.at(i);
        FloatRect fragmentRect(fragment.x, fragment.y - baseline, fragment.width, fragment.height);
        fragment.buildFragmentTransform(fragmentTransform);
        if (!fragmentTransform.isIdentity())
            fragmentRect = fragmentTransform.mapRect(fragmentRect);

        textRect.unite(fragmentRect);
    }

    return textRect;
}

void SVGInlineTextBox::setTextFragments(Vector<SVGTextFragment>&& fragments)
{
    m_textFragments = WTFMove(fragments);
}

} // namespace WebCore
