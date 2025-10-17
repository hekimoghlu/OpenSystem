/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#include "PaintRenderingContext2D.h"

#include "CustomPaintCanvas.h"
#include "DisplayListDrawingContext.h"
#include "DisplayListRecorder.h"
#include "DisplayListReplayer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PaintRenderingContext2D);

std::unique_ptr<PaintRenderingContext2D> PaintRenderingContext2D::create(CustomPaintCanvas& canvas)
{
    return std::unique_ptr<PaintRenderingContext2D>(new PaintRenderingContext2D(canvas));
}

PaintRenderingContext2D::PaintRenderingContext2D(CustomPaintCanvas& canvas)
    : CanvasRenderingContext2DBase(canvas, Type::Paint, { }, false)
{
}

PaintRenderingContext2D::~PaintRenderingContext2D() = default;

CustomPaintCanvas& PaintRenderingContext2D::canvas() const
{
    return downcast<CustomPaintCanvas>(canvasBase());
}

GraphicsContext* PaintRenderingContext2D::ensureDrawingContext() const
{
    if (!m_recordingContext)
        m_recordingContext = makeUnique<DisplayList::DrawingContext>(canvasBase().size());
    return &m_recordingContext->context();
}

GraphicsContext* PaintRenderingContext2D::existingDrawingContext() const
{
    return m_recordingContext ? &m_recordingContext->context() : nullptr;
}

AffineTransform PaintRenderingContext2D::baseTransform() const
{
    // The base transform of the display list.
    // FIXME: this is actually correct, but the display list will not behave correctly with respect to
    // playback. The GraphicsContext should be fixed to start at identity transform, and the
    // device transform should be a separate concept that the display list or context2d cannot reset.
    return { };
}

void PaintRenderingContext2D::replayDisplayList(GraphicsContext& target) const
{
    if (!m_recordingContext)
        return;
    auto& displayList = m_recordingContext->displayList();
    if (displayList.isEmpty())
        return;
    DisplayList::Replayer replayer(target, displayList);
    replayer.replay(FloatRect { { }, canvasBase().size() });
    displayList.clear();
}

} // namespace WebCore
