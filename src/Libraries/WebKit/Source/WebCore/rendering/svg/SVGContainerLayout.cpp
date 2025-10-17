/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#include "SVGContainerLayout.h"

#include "Logging.h"
#include "RenderAncestorIterator.h"
#include "RenderChildIterator.h"
#include "RenderSVGInline.h"
#include "RenderSVGModelObject.h"
#include "RenderSVGResourceGradient.h"
#include "RenderSVGRoot.h"
#include "RenderSVGShape.h"
#include "RenderSVGText.h"
#include "RenderSVGTransformableContainer.h"
#include "RenderSVGViewportContainer.h"
#include "SVGRenderSupport.h"

namespace WebCore {

SVGContainerLayout::SVGContainerLayout(RenderLayerModelObject& container)
    : m_container(container)
{
}

void SVGContainerLayout::layoutChildren(bool containerNeedsLayout)
{
    bool layoutSizeChanged = layoutSizeOfNearestViewportChanged();
    bool transformChanged = transformToRootChanged(m_container.ptr());

    m_positionedChildren.clear();
    for (auto& child : childrenOfType<RenderObject>(m_container)) {
        if (child.isSVGLayerAwareRenderer())
            m_positionedChildren.append(downcast<RenderLayerModelObject>(child));

        bool needsLayout = containerNeedsLayout;
        bool childEverHadLayout = child.everHadLayout();

        if (transformChanged) {
            // If the transform changed we need to update the text metrics (note: this also happens for layoutSizeChanged=true).
            if (CheckedPtr text = dynamicDowncast<RenderSVGText>(child))
                text->setNeedsTextMetricsUpdate();
            needsLayout = true;
        }

        if (layoutSizeChanged) {
            if (child.isAnonymous()) {
                ASSERT(is<RenderSVGViewportContainer>(child));
                needsLayout = true;
            } else if (RefPtr element = dynamicDowncast<SVGElement>(child.node())) {
                if (element->hasRelativeLengths()) {
                    // When containerNeedsLayout is false and the layout size changed, we have to check whether this child uses relative lengths

                    // When the layout size changed and when using relative values tell the RenderSVGShape to update its shape object
                    if (CheckedPtr shape = dynamicDowncast<RenderSVGShape>(child)) {
                        shape->setNeedsShapeUpdate();
                        needsLayout = true;
                    } else if (CheckedPtr svgText = dynamicDowncast<RenderSVGText>(child)) {
                        svgText->setNeedsTextMetricsUpdate();
                        svgText->setNeedsPositioningValuesUpdate();
                        needsLayout = true;
                    } else if (CheckedPtr resource = dynamicDowncast<RenderSVGResourceGradient>(child))
                        resource->invalidateGradient();
                    // FIXME: [LBSE] Add pattern support.
                }
            }
        }

        if (needsLayout)
            child.setNeedsLayout(MarkOnlyThis);

        if (CheckedPtr element = dynamicDowncast<RenderElement>(child)) {
            if (element->needsLayout())
                element->layout();

            if (!childEverHadLayout && element->checkForRepaintDuringLayout())
                element->repaint();
        }

        ASSERT(!child.needsLayout());
    }
}

void SVGContainerLayout::positionChildrenRelativeToContainer()
{
    if (m_positionedChildren.isEmpty())
        return;

    auto verifyPositionedChildRendererExpectation = [](RenderObject& renderer) {
#if !defined(NDEBUG)
        ASSERT(renderer.isSVGLayerAwareRenderer()); // Pre-condition to enter m_positionedChildren
        ASSERT(!renderer.isRenderSVGRoot()); // There is only one outermost RenderSVGRoot object
        ASSERT(!renderer.isRenderSVGInline()); // Inlines are only allowed within a RenderSVGText tree

        if (is<RenderSVGModelObject>(renderer) || is<RenderSVGBlock>(renderer))
            return;

        ASSERT_NOT_REACHED();
        return;
#else
        UNUSED_PARAM(renderer);
#endif
    };

    auto computeContainerLayoutLocation = [&]() -> LayoutPoint {
        // The nominal SVG layout location (== flooredLayoutPoint(objectBoundingBoxWithoutTransformsTopLeft), where
        // objectBoundingBoxWithoutTransforms = union of child boxes, not mapped through their tranforms) is
        // only meaningful for the children of the RenderSVGRoot. RenderSVGRoot itself is positioned according to
        // the CSS box model object, where we need to respect border & padding, encoded in the contentBoxLocation().
        // -> Position all RenderSVGRoot children relative to the contentBoxLocation() to avoid intruding border/padding area.
        if (CheckedPtr svgRoot = dynamicDowncast<RenderSVGRoot>(m_container.get()))
            return -svgRoot->contentBoxLocation();

        // For (inner) RenderSVGViewportContainer nominalSVGLayoutLocation() returns the viewport boundaries,
        // including the effect of the 'x'/'y' attribute values. Do not subtract the location, otherwise the
        // effect of the x/y translation is removed.
        if (is<RenderSVGViewportContainer>(m_container) && !m_container->isAnonymous())
            return { };

        return m_container->nominalSVGLayoutLocation();
    };

    // Arrange layout location for all child renderers relative to the container layout location.
    auto parentLayoutLocation = computeContainerLayoutLocation();
    for (RenderLayerModelObject& child : m_positionedChildren) {
        verifyPositionedChildRendererExpectation(child);

        auto desiredLayoutLocation = toLayoutPoint(child.nominalSVGLayoutLocation() - parentLayoutLocation);
        if (child.currentSVGLayoutLocation() != desiredLayoutLocation)
            child.setCurrentSVGLayoutLocation(desiredLayoutLocation);
    }
}

void SVGContainerLayout::verifyLayoutLocationConsistency(const RenderLayerModelObject& renderer)
{
    if (renderer.isSVGLayerAwareRenderer() && !renderer.isRenderSVGRoot()) {
        auto currentLayoutLocation = renderer.currentSVGLayoutLocation();

        auto expectedLayoutLocation = currentLayoutLocation;
        for (auto& ancestor : ancestorsOfType<RenderLayerModelObject>(renderer)) {
            ASSERT(ancestor.isSVGLayerAwareRenderer());
            if (ancestor.isRenderSVGRoot())
                break;
            expectedLayoutLocation.moveBy(ancestor.currentSVGLayoutLocation());
        }

        auto initialLayoutLocation = renderer.nominalSVGLayoutLocation();
        if (expectedLayoutLocation == initialLayoutLocation) {
            LOG_WITH_STREAM(SVG, stream << "--> SVGContainerLayout renderer " << &renderer << " (" << renderer.renderName().characters() << ")"
                << " - verifyLayoutLocationConsistency() currentSVGLayoutLocation / nominalSVGLayoutLocation are in sync.");
        } else {
            LOG_WITH_STREAM(SVG, stream << "--> SVGContainerLayout renderer " << &renderer << " (" << renderer.renderName().characters() << ")"
                << " - verifyLayoutLocationConsistency() currentSVGLayoutLocation / nominalSVGLayoutLocation invariant violated -- out of sync due to partial layout?"
                << " currentLayoutLocation=" << currentLayoutLocation
                << "  (expectedLayoutLocation=" << expectedLayoutLocation
                << " != initialLayoutLocation=" << initialLayoutLocation << ")"
                << " -> aborting with a render tree dump");

#if ENABLE(TREE_DEBUGGING)
            showRenderTree(&renderer);
#endif

            ASSERT_NOT_REACHED();
        }
    }

    for (auto& child : childrenOfType<RenderLayerModelObject>(renderer)) {
        if (child.isSVGLayerAwareRenderer())
            verifyLayoutLocationConsistency(child);
    }

#if !defined(NDEBUG)
    if (renderer.isRenderSVGRoot()) {
        LOG_WITH_STREAM(SVG, stream << "--> SVGContainerLayout renderer " << &renderer << " (" << renderer.renderName().characters() << ")"
            << " - verifyLayoutLocationConsistency() end");
    }
#endif
}

bool SVGContainerLayout::layoutSizeOfNearestViewportChanged() const
{
    RenderElement* ancestor = m_container.ptr();
    while (ancestor && !is<RenderSVGRoot>(ancestor) && !is<RenderSVGViewportContainer>(ancestor))
        ancestor = ancestor->parent();

    ASSERT(ancestor);
    if (auto* viewportContainer = dynamicDowncast<RenderSVGViewportContainer>(ancestor))
        return viewportContainer->isLayoutSizeChanged();

    if (auto* svgRoot = dynamicDowncast<RenderSVGRoot>(ancestor))
        return svgRoot->isLayoutSizeChanged();

    return false;
}

bool SVGContainerLayout::transformToRootChanged(const RenderObject* ancestor)
{
    while (ancestor) {
        if (CheckedPtr container = dynamicDowncast<RenderSVGTransformableContainer>(*ancestor))
            return container->didTransformToRootUpdate();

        if (CheckedPtr container = dynamicDowncast<RenderSVGViewportContainer>(*ancestor))
            return container->didTransformToRootUpdate();

        if (CheckedPtr svgRoot = dynamicDowncast<RenderSVGRoot>(*ancestor))
            return svgRoot->didTransformToRootUpdate();
        ancestor = ancestor->parent();
    }

    return false;
}

}

