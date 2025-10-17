/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

#include "RenderBoxModelObject.h"

namespace WebCore {

class GraphicsContext;
class FloatRoundedRect;

enum class ShadowStyle : bool;

class FloatRoundedRect;

enum BaseBackgroundColorUsage {
    BaseBackgroundColorUse,
    BaseBackgroundColorOnly,
    BaseBackgroundColorSkip
};

struct BackgroundImageGeometry {
    BackgroundImageGeometry(const LayoutRect& destinationRect, const LayoutSize& tileSizeWithoutPixelSnapping, const LayoutSize& tileSize, const LayoutSize& phase, const LayoutSize& spaceSize, bool fixedAttachment);

    LayoutSize relativePhase() const
    {
        LayoutSize relativePhase = phase;
        relativePhase += destinationRect.location() - destinationOrigin;
        return relativePhase;
    }

    void clip(const LayoutRect& clipRect) { destinationRect.intersect(clipRect); }

    LayoutRect destinationRect;
    LayoutPoint destinationOrigin;
    LayoutSize tileSizeWithoutPixelSnapping;
    LayoutSize tileSize;
    LayoutSize phase;
    LayoutSize spaceSize;
    bool hasNonLocalGeometry; // Has background-attachment: fixed. Implies that we can't always cheaply compute destRect.
};

class BackgroundPainter {
public:
    BackgroundPainter(RenderBoxModelObject&, const PaintInfo&);

    void setOverrideClip(FillBox overrideClip) { m_overrideClip = overrideClip; }
    void setOverrideOrigin(FillBox overrideOrigin) { m_overrideOrigin = overrideOrigin; }

    void paintBackground(const LayoutRect&, BleedAvoidance) const;

    void paintFillLayers(const Color&, const FillLayer&, const LayoutRect&, BleedAvoidance, CompositeOperator, RenderElement* backgroundObject = nullptr) const;
    void paintFillLayer(const Color&, const FillLayer&, const LayoutRect&, BleedAvoidance, const InlineIterator::InlineBoxIterator&, const LayoutRect& backgroundImageStrip = { }, CompositeOperator = CompositeOperator::SourceOver, RenderElement* backgroundObject = nullptr, BaseBackgroundColorUsage = BaseBackgroundColorUse) const;

    void paintBoxShadow(const LayoutRect&, const RenderStyle&, ShadowStyle, RectEdges<bool> closedEdges = { true, true, true, true }) const;

    static bool paintsOwnBackground(const RenderBoxModelObject&);
    static BackgroundImageGeometry calculateBackgroundImageGeometry(const RenderBoxModelObject&, const RenderLayerModelObject* paintContainer, const FillLayer&, const LayoutPoint& paintOffset, const LayoutRect& borderBoxRect, std::optional<FillBox> overrideOrigin = std::nullopt);
    static void clipRoundedInnerRect(GraphicsContext&, const FloatRoundedRect& clipRect);
    static bool boxShadowShouldBeAppliedToBackground(const RenderBoxModelObject&, const LayoutPoint& paintOffset, BleedAvoidance, const InlineIterator::InlineBoxIterator&);

private:
    void paintRootBoxFillLayers() const;

    static LayoutSize calculateFillTileSize(const RenderBoxModelObject&, const FillLayer&, const LayoutSize& positioningAreaSize);

    const Document& document() const;
    const RenderView& view() const;

    RenderBoxModelObject& m_renderer;
    const PaintInfo& m_paintInfo;
    std::optional<FillBox> m_overrideClip;
    std::optional<FillBox> m_overrideOrigin;
};

}
