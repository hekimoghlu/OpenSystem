/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include "ResizeObservation.h"

#include "ElementInlines.h"
#include "HTMLFrameOwnerElement.h"
#include "Logging.h"
#include "RenderBoxInlines.h"
#include "RenderElementInlines.h"
#include "SVGElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

Ref<ResizeObservation> ResizeObservation::create(Element& target, ResizeObserverBoxOptions observedBox)
{
    return adoptRef(*new ResizeObservation(target, observedBox));
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ResizeObservation);

ResizeObservation::ResizeObservation(Element& element, ResizeObserverBoxOptions observedBox)
    : m_target { element }
    , m_lastObservationSizes { LayoutSize(-1, -1), LayoutSize(-1, -1), LayoutSize(-1, -1) }
    , m_observedBox { observedBox }
{
}

ResizeObservation::~ResizeObservation() = default;

void ResizeObservation::updateObservationSize(const BoxSizes& boxSizes)
{
    m_lastObservationSizes = boxSizes;
}

void ResizeObservation::resetObservationSize()
{
    m_lastObservationSizes = { LayoutSize(-1, -1), LayoutSize(-1, -1), LayoutSize(-1, -1) };
}

auto ResizeObservation::computeObservedSizes() const -> std::optional<BoxSizes>
{
    if (auto* svg = dynamicDowncast<SVGElement>(target())) {
        if (svg->hasAssociatedSVGLayoutBox()) {
            LayoutSize size;
            if (auto svgRect = svg->getBoundingBox()) {
                size.setWidth(svgRect->width());
                size.setHeight(svgRect->height());
            }
            return { { size, size, size } };
        }
    }

    auto* box = m_target->renderBox();
    if (box) {
        if (box->isSkippedContent())
            return std::nullopt;
        return { {
            adjustLayoutSizeForAbsoluteZoom(box->contentBoxSize(), *box),
            adjustLayoutSizeForAbsoluteZoom(box->contentBoxLogicalSize(), *box),
            adjustLayoutSizeForAbsoluteZoom(box->borderBoxLogicalSize(), *box)
        } };
    }

    return BoxSizes { };
}

LayoutPoint ResizeObservation::computeTargetLocation() const
{
    if (!m_target->isSVGElement()) {
        if (auto box = m_target->renderBox())
            return LayoutPoint(box->paddingLeft(), box->paddingTop());
    }

    return { };
}

FloatRect ResizeObservation::computeContentRect() const
{
    return FloatRect(FloatPoint(computeTargetLocation()), FloatSize(m_lastObservationSizes.contentBoxSize));
}

FloatSize ResizeObservation::borderBoxSize() const
{
    return m_lastObservationSizes.borderBoxLogicalSize;
}

FloatSize ResizeObservation::contentBoxSize() const
{
    return m_lastObservationSizes.contentBoxLogicalSize;
}

FloatSize ResizeObservation::snappedContentBoxSize() const
{
    return m_lastObservationSizes.contentBoxLogicalSize; // FIXME: Need to pixel snap.
}

RefPtr<Element> ResizeObservation::protectedTarget() const
{
    return m_target.get();
}

std::optional<ResizeObservation::BoxSizes> ResizeObservation::elementSizeChanged() const
{
    auto currentSizes = computeObservedSizes();
    if (!currentSizes)
        return std::nullopt;

    LOG_WITH_STREAM(ResizeObserver, stream << "ResizeObservation " << this << " elementSizeChanged - new content box " << currentSizes->contentBoxSize);

    switch (m_observedBox) {
    case ResizeObserverBoxOptions::BorderBox:
        if (m_lastObservationSizes.borderBoxLogicalSize != currentSizes->borderBoxLogicalSize)
            return currentSizes;
        break;
    case ResizeObserverBoxOptions::ContentBox:
        if (m_lastObservationSizes.contentBoxLogicalSize != currentSizes->contentBoxLogicalSize)
            return currentSizes;
        break;
    }

    return { };
}

// https://drafts.csswg.org/resize-observer/#calculate-depth-for-node
size_t ResizeObservation::targetElementDepth() const
{
    unsigned depth = 0;
    for (Element* ownerElement = m_target.get(); ownerElement; ownerElement = ownerElement->document().ownerElement()) {
        for (Element* parent = ownerElement; parent; parent = parent->parentElementInComposedTree())
            ++depth;
    }

    return depth;
}

TextStream& operator<<(TextStream& ts, const ResizeObservation& observation)
{
    ts.dumpProperty("target", ValueOrNull(observation.target()));
    ts.dumpProperty("border box", observation.borderBoxSize());
    ts.dumpProperty("content box", observation.contentBoxSize());
    ts.dumpProperty("snapped content box", observation.snappedContentBoxSize());
    return ts;
}

} // namespace WebCore
