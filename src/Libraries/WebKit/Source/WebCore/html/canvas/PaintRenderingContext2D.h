/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

#include "CanvasRenderingContext2DBase.h"

namespace WebCore {

namespace DisplayList {
class DrawingContext;
}

class CustomPaintCanvas;

class PaintRenderingContext2D final : public CanvasRenderingContext2DBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PaintRenderingContext2D);
public:
    static std::unique_ptr<PaintRenderingContext2D> create(CustomPaintCanvas&);

    virtual ~PaintRenderingContext2D();

    GraphicsContext* ensureDrawingContext() const;
    GraphicsContext* existingDrawingContext() const final;
    AffineTransform baseTransform() const final;

    CustomPaintCanvas& canvas() const;
    void replayDisplayList(GraphicsContext& target) const;

private:
    PaintRenderingContext2D(CustomPaintCanvas&);
    mutable std::unique_ptr<DisplayList::DrawingContext> m_recordingContext;
};

} // namespace WebCore
SPECIALIZE_TYPE_TRAITS_CANVASRENDERINGCONTEXT(WebCore::PaintRenderingContext2D, isPaint())
