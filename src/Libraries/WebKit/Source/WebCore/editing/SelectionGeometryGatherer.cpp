/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#include "SelectionGeometryGatherer.h"

#if ENABLE(SERVICE_CONTROLS)

#include "Editor.h"
#include "EditorClient.h"
#include "ImageOverlayController.h"
#include "LocalFrame.h"
#include "RenderView.h"
#include "ServicesOverlayController.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SelectionGeometryGatherer);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SelectionGeometryGatherer::Notifier);

SelectionGeometryGatherer::SelectionGeometryGatherer(RenderView& renderView)
    : m_renderView(renderView)
    , m_isTextOnly(true)
{
}

void SelectionGeometryGatherer::addQuad(const RenderLayerModelObject* repaintContainer, const FloatQuad& quad)
{
    if (!quad.boundingBoxIsEmpty())
        m_quads.append(repaintContainer ? repaintContainer->localToAbsoluteQuad(quad) : quad);
}

void SelectionGeometryGatherer::addGapRects(const RenderLayerModelObject* repaintContainer, const GapRects& rects)
{
    if (repaintContainer) {
        GapRects absoluteGapRects;
        absoluteGapRects.uniteLeft(LayoutRect(repaintContainer->localToAbsoluteQuad(FloatQuad(rects.left())).boundingBox()));
        absoluteGapRects.uniteCenter(LayoutRect(repaintContainer->localToAbsoluteQuad(FloatQuad(rects.center())).boundingBox()));
        absoluteGapRects.uniteRight(LayoutRect(repaintContainer->localToAbsoluteQuad(FloatQuad(rects.right())).boundingBox()));
        m_gapRects.append(absoluteGapRects);
    } else
        m_gapRects.append(rects);
}

SelectionGeometryGatherer::Notifier::Notifier(SelectionGeometryGatherer& gatherer)
    : m_gatherer(gatherer)
{
}

SelectionGeometryGatherer::Notifier::~Notifier()
{
    RefPtr page = m_gatherer.m_renderView->view().frame().page();
    if (!page)
        return;

    page->protectedServicesOverlayController()->selectionRectsDidChange(m_gatherer.boundingRects(), m_gatherer.m_gapRects, m_gatherer.isTextOnly());
    page->imageOverlayController().selectionQuadsDidChange(m_gatherer.m_renderView->frame(), m_gatherer.m_quads);
}

Vector<LayoutRect> SelectionGeometryGatherer::boundingRects() const
{
    return m_quads.map([](auto& quad) {
        return LayoutRect { quad.boundingBox() };
    });
}

std::unique_ptr<SelectionGeometryGatherer::Notifier> SelectionGeometryGatherer::clearAndCreateNotifier()
{
    m_quads.clear();
    m_gapRects.clear();
    m_isTextOnly = true;

    return makeUnique<Notifier>(*this);
}

} // namespace WebCore

#endif // ENABLE(SERVICE_CONTROLS)
