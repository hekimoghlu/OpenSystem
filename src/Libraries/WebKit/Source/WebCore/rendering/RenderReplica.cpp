/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#include "RenderReplica.h"

#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayer.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderReplica);

RenderReplica::RenderReplica(Document& document, RenderStyle&& style)
    : RenderBox(Type::Replica, document, WTFMove(style))
{
    // This is a hack. Replicas are synthetic, and don't pick up the attributes of the
    // renderers being replicated, so they always report that they are inline, non-replaced.
    // However, we need transforms to be applied to replicas for reflections, so have to pass
    // the if (!isInline() || isReplaced()) check before setHasTransform().
    // FIXME: Is the comment above obsolete? Can't find a check of isReplacedOrAtomicInline guarding setHasTransform any more.
    setReplacedOrAtomicInline(true);
}

RenderReplica::~RenderReplica() = default;
    
void RenderReplica::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    setFrameRect(parentBox()->borderBoxRect());
    updateLayerTransform();
    clearNeedsLayout();
}

void RenderReplica::computePreferredLogicalWidths()
{
    m_minPreferredLogicalWidth = parentBox()->width();
    m_maxPreferredLogicalWidth = m_minPreferredLogicalWidth;
    setPreferredLogicalWidthsDirty(false);
}

void RenderReplica::paint(PaintInfo& paintInfo, const LayoutPoint& paintOffset)
{
    if (paintInfo.phase != PaintPhase::Foreground && paintInfo.phase != PaintPhase::Mask)
        return;
 
    LayoutPoint adjustedPaintOffset = paintOffset + location();

    if (paintInfo.phase == PaintPhase::Foreground) {
        // Turn around and paint the parent layer. Use temporary clipRects, so that the layer doesn't end up caching clip rects
        // computing using the wrong rootLayer
        RenderLayer* rootPaintingLayer = layer()->transform() ? layer()->parent() : layer()->enclosingTransformedAncestor();
        RenderLayer::LayerPaintingInfo paintingInfo(rootPaintingLayer, paintInfo.rect, PaintBehavior::Normal, LayoutSize(), 0);
        OptionSet<RenderLayer::PaintLayerFlag> flags { RenderLayer::PaintLayerFlag::HaveTransparency, RenderLayer::PaintLayerFlag::AppliedTransform, RenderLayer::PaintLayerFlag::TemporaryClipRects, RenderLayer::PaintLayerFlag::PaintingReflection };
        layer()->parent()->paintLayer(paintInfo.context(), paintingInfo, flags);
    } else if (paintInfo.phase == PaintPhase::Mask)
        paintMask(paintInfo, adjustedPaintOffset);
}

} // namespace WebCore
