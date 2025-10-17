/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#include "AccessibilityRegionContext.h"
#include "AffineTransform.h"
#include "EventRegion.h"
#include "GraphicsContext.h"
#include "IntRect.h"
#include "LayoutRect.h"
#include "PaintPhase.h"
#include <limits>
#include <wtf/HashMap.h>
#include <wtf/OptionSet.h>
#include <wtf/WeakListHashSet.h>

namespace WebCore {

class OverlapTestRequestClient;
class RenderInline;
class RenderLayer;
class RenderLayerModelObject;
class RenderObject;

typedef UncheckedKeyHashMap<OverlapTestRequestClient*, IntRect> OverlapTestRequestMap;

/*
 * Paint the object and its children, clipped by (x|y|w|h).
 * (tx|ty) is the calculated position of the parent
 */
struct PaintInfo {
    PaintInfo(GraphicsContext& newContext, const LayoutRect& newRect, PaintPhase newPhase, OptionSet<PaintBehavior> newPaintBehavior,
        RenderObject* newSubtreePaintRoot = nullptr, SingleThreadWeakListHashSet<RenderInline>* newOutlineObjects = nullptr,
        OverlapTestRequestMap* overlapTestRequests = nullptr, const RenderLayerModelObject* newPaintContainer = nullptr,
        const RenderLayer* enclosingSelfPaintingLayer = nullptr, bool newRequireSecurityOriginAccessForWidgets = false)
            : rect(newRect)
            , phase(newPhase)
            , paintBehavior(newPaintBehavior)
            , subtreePaintRoot(newSubtreePaintRoot)
            , outlineObjects(newOutlineObjects)
            , overlapTestRequests(overlapTestRequests)
            , paintContainer(newPaintContainer)
            , requireSecurityOriginAccessForWidgets(newRequireSecurityOriginAccessForWidgets)
            , m_enclosingSelfPaintingLayer(enclosingSelfPaintingLayer)
            , m_context(&newContext)
    {
    }

    GraphicsContext& context() const
    {
        ASSERT(m_context);
        return *m_context;
    }

    void setContext(GraphicsContext& context)
    {
        m_context = &context;
    }

    void updateSubtreePaintRootForChildren(const RenderObject* renderer)
    {
        if (!subtreePaintRoot)
            return;

        // If we're the painting root, kids draw normally, and see root of nullptr.
        if (subtreePaintRoot == renderer) {
            subtreePaintRoot = nullptr;
            return;
        }
    }

    bool shouldPaintWithinRoot(const RenderObject& renderer) const
    {
        return !subtreePaintRoot || subtreePaintRoot == &renderer;
    }

    bool forceTextColor() const { return forceBlackText() || forceWhiteText(); }
    bool forceBlackText() const { return paintBehavior.contains(PaintBehavior::ForceBlackText); }
    bool forceWhiteText() const { return paintBehavior.contains(PaintBehavior::ForceWhiteText); }
    Color forcedTextColor() const { return forceBlackText() ? Color::black : Color::white; }

    bool skipRootBackground() const { return paintBehavior.contains(PaintBehavior::SkipRootBackground); }
    bool paintRootBackgroundOnly() const { return paintBehavior.contains(PaintBehavior::RootBackgroundOnly); }

    const RenderLayer* enclosingSelfPaintingLayer() const { return m_enclosingSelfPaintingLayer; }

    void applyTransform(const AffineTransform& localToAncestorTransform)
    {
        if (localToAncestorTransform.isIdentity())
            return;

        context().concatCTM(localToAncestorTransform);

        if (rect.isInfinite())
            return;

        FloatRect tranformedRect(valueOrDefault(localToAncestorTransform.inverse()).mapRect(rect));
        rect.setLocation(LayoutPoint(tranformedRect.location()));
        rect.setSize(LayoutSize(tranformedRect.size()));
    }

    EventRegionContext* eventRegionContext() { return dynamicDowncast<EventRegionContext>(regionContext); }
    AccessibilityRegionContext* accessibilityRegionContext() { return dynamicDowncast<AccessibilityRegionContext>(regionContext); }

    LayoutRect rect;
    PaintPhase phase;
    OptionSet<PaintBehavior> paintBehavior;
    RenderObject* subtreePaintRoot; // used to draw just one element and its visual children
    SingleThreadWeakListHashSet<RenderInline>* outlineObjects; // used to list outlines that should be painted by a block with inline children
    OverlapTestRequestMap* overlapTestRequests;
    const RenderLayerModelObject* paintContainer; // the layer object that originates the current painting
    bool requireSecurityOriginAccessForWidgets { false };
    const RenderLayer* m_enclosingSelfPaintingLayer { nullptr };
    RegionContext* regionContext { nullptr }; // For PaintPhase::EventRegion and PaintPhase::Accessibility.

private:
    GraphicsContext* m_context;
};

} // namespace WebCore
