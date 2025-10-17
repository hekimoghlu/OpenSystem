/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#if ENABLE(OFFSCREEN_CANVAS)

#include "ActiveDOMObject.h"
#include "AffineTransform.h"
#include "CanvasBase.h"
#include "ContextDestructionObserver.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "IDLTypes.h"
#include "ImageBuffer.h"
#include "IntSize.h"
#include "ScriptWrappable.h"
#include <wtf/FixedVector.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if ENABLE(WEBGL)
#include "WebGLContextAttributes.h"
#endif

namespace WebCore {

class CanvasRenderingContext;
class DeferredPromise;
class GPU;
class GPUCanvasContext;
class HTMLCanvasElement;
class ImageBitmap;
class ImageBitmapRenderingContext;
class ImageData;
class OffscreenCanvasRenderingContext2D;
class WebGL2RenderingContext;
class WebGLRenderingContext;
class WebGLRenderingContextBase;

using OffscreenRenderingContext = std::variant<
#if ENABLE(WEBGL)
    RefPtr<WebGLRenderingContext>,
    RefPtr<WebGL2RenderingContext>,
#endif
    RefPtr<GPUCanvasContext>,
    RefPtr<ImageBitmapRenderingContext>,
    RefPtr<OffscreenCanvasRenderingContext2D>
>;

class PlaceholderRenderingContext;
class PlaceholderRenderingContextSource;

class DetachedOffscreenCanvas {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(DetachedOffscreenCanvas, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(DetachedOffscreenCanvas);
    friend class OffscreenCanvas;

public:
    DetachedOffscreenCanvas(const IntSize&, bool originClean, RefPtr<PlaceholderRenderingContextSource>&&);
    WEBCORE_EXPORT ~DetachedOffscreenCanvas();
    const IntSize& size() const { return m_size; }
    bool originClean() const { return m_originClean; }
    RefPtr<PlaceholderRenderingContextSource> takePlaceholderSource();

private:
    RefPtr<PlaceholderRenderingContextSource> m_placeholderSource;
    IntSize m_size;
    bool m_originClean;
};

class OffscreenCanvas final : public ActiveDOMObject, public RefCounted<OffscreenCanvas>, public CanvasBase, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(OffscreenCanvas, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    struct ImageEncodeOptions {
        String type = "image/png"_s;
        double quality = 1.0;
    };

    enum class RenderingContextType {
        _2d,
        Webgl,
        Webgl2,
        Bitmaprenderer,
        Webgpu
    };

    static bool enabledForContext(ScriptExecutionContext&);

    static Ref<OffscreenCanvas> create(ScriptExecutionContext&, unsigned width, unsigned height);
    static Ref<OffscreenCanvas> create(ScriptExecutionContext&, std::unique_ptr<DetachedOffscreenCanvas>&&);
    static Ref<OffscreenCanvas> create(ScriptExecutionContext&, PlaceholderRenderingContext&);
    WEBCORE_EXPORT virtual ~OffscreenCanvas();

    void setWidth(unsigned);
    void setHeight(unsigned);

    void setImageBufferAndMarkDirty(RefPtr<ImageBuffer>&&) final;

    CanvasRenderingContext* renderingContext() const final { return m_context.get(); }

    const CSSParserContext& cssParserContext() const final;

    ExceptionOr<std::optional<OffscreenRenderingContext>> getContext(JSC::JSGlobalObject&, RenderingContextType, FixedVector<JSC::Strong<JSC::Unknown>>&& arguments);
    ExceptionOr<RefPtr<ImageBitmap>> transferToImageBitmap();
    void convertToBlob(ImageEncodeOptions&&, Ref<DeferredPromise>&&);

    void didDraw(const std::optional<FloatRect>&, ShouldApplyPostProcessingToDirtyRect) final;

    Image* copiedImage() const final;
    void clearCopiedImage() const final;

    SecurityOrigin* securityOrigin() const final;

    bool canDetach() const;
    std::unique_ptr<DetachedOffscreenCanvas> detach();

    void commitToPlaceholderCanvas();

    void queueTaskKeepingObjectAlive(TaskSource, Function<void()>&&) final;
    void dispatchEvent(Event&) final;
    bool isDetached() const { return m_detached; };

private:
    OffscreenCanvas(ScriptExecutionContext&, IntSize, RefPtr<PlaceholderRenderingContextSource>&&);

    bool isOffscreenCanvas() const final { return true; }

    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }
    ScriptExecutionContext* canvasBaseScriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::OffscreenCanvas; }
    void refEventTarget() final { RefCounted::ref(); }
    void derefEventTarget() final { RefCounted::deref(); }

    void setSize(const IntSize&) final;

    void createImageBuffer() const final;

    void reset();
    void scheduleCommitToPlaceholderCanvas();

    std::unique_ptr<CanvasRenderingContext> m_context;
    RefPtr<PlaceholderRenderingContextSource> m_placeholderSource;
    mutable RefPtr<Image> m_copiedImage;
    bool m_detached { false };
    bool m_hasScheduledCommit { false };

    mutable std::unique_ptr<CSSParserContext> m_cssParserContext;
};

}

SPECIALIZE_TYPE_TRAITS_CANVAS(WebCore::OffscreenCanvas, isOffscreenCanvas())

#endif
