/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

#include "AffineTransform.h"
#include "CanvasBase.h"
#include "ContextDestructionObserver.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "ImageBuffer.h"
#include "IntSize.h"
#include "PaintRenderingContext2D.h"
#include "ScriptWrappable.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ImageBitmap;

namespace DisplayList {
class DrawingContext;
}

class CustomPaintCanvas final : public RefCounted<CustomPaintCanvas>, public CanvasBase, private ContextDestructionObserver {
    WTF_MAKE_TZONE_ALLOCATED(CustomPaintCanvas);
public:

    static Ref<CustomPaintCanvas> create(ScriptExecutionContext&, unsigned width, unsigned height);
    virtual ~CustomPaintCanvas();
    bool isCustomPaintCanvas() const final { return true; }

    RefPtr<PaintRenderingContext2D> getContext();

    CanvasRenderingContext* renderingContext() const final { return m_context.get(); }

    void didDraw(const std::optional<FloatRect>&, ShouldApplyPostProcessingToDirtyRect) final { }

    Image* copiedImage() const final;
    void clearCopiedImage() const final;

    void replayDisplayList(GraphicsContext&);

    void queueTaskKeepingObjectAlive(TaskSource, Function<void()>&&) final { };
    void dispatchEvent(Event&) final { }

    const CSSParserContext& cssParserContext() const final;

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    CustomPaintCanvas(ScriptExecutionContext&, unsigned width, unsigned height);

    ScriptExecutionContext* canvasBaseScriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

    std::unique_ptr<PaintRenderingContext2D> m_context;
    mutable RefPtr<Image> m_copiedImage;
    mutable std::unique_ptr<CSSParserContext> m_cssParserContext;
};

}
SPECIALIZE_TYPE_TRAITS_CANVAS(WebCore::CustomPaintCanvas, isCustomPaintCanvas())
