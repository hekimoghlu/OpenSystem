/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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

#include "CanvasBase.h"
#include "GraphicsLayerContentsDisplayDelegate.h"
#include "ImageBuffer.h"
#include "ScriptWrappable.h"
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class CSSStyleImageValue;
class CachedImage;
class CanvasPattern;
class DestinationColorSpace;
class GraphicsLayer;
class HTMLCanvasElement;
class HTMLImageElement;
class HTMLVideoElement;
class ImageBitmap;
class SVGImageElement;
class WebGLObject;
enum class ImageBufferPixelFormat : uint8_t;

class CanvasRenderingContext : public ScriptWrappable, public CanMakeWeakPtr<CanvasRenderingContext> {
    WTF_MAKE_NONCOPYABLE(CanvasRenderingContext);
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CanvasRenderingContext);
public:
    virtual ~CanvasRenderingContext();

    static UncheckedKeyHashSet<CanvasRenderingContext*>& instances() WTF_REQUIRES_LOCK(instancesLock());
    static Lock& instancesLock() WTF_RETURNS_LOCK(s_instancesLock);

    WEBCORE_EXPORT void ref() const;
    WEBCORE_EXPORT void deref() const;

    CanvasBase& canvasBase() const { return m_canvas; }

    bool is2dBase() const { return is2d() || isOffscreen2d() || isPaint(); }
    bool is2d() const { return m_type == Type::CanvasElement2D; }
    bool isWebGL1() const { return m_type == Type::WebGL1; }
    bool isWebGL2() const { return m_type == Type::WebGL2; }
    bool isWebGL() const { return isWebGL1() || isWebGL2(); }
    bool isWebGPU() const { return m_type == Type::WebGPU; }
    bool isGPUBased() const { return isWebGPU() || isWebGL(); }
    bool isBitmapRenderer() const { return m_type == Type::BitmapRenderer; }
    bool isPlaceholder() const { return m_type == Type::Placeholder; }
    bool isOffscreen2d() const { return m_type == Type::Offscreen2D; }
    bool isPaint() const { return m_type == Type::Paint; }

    virtual void clearAccumulatedDirtyRect() { }

    // Canvas 2DContext drawing buffer is the same as display buffer.
    // WebGL, WebGPU draws to drawing buffer. The draw buffer is then swapped to
    // display buffer during preparation and compositor composites the display buffer.
    // toDataURL and similar functions from JS execution reads the drawing buffer.
    // Web Inspector and similar reads from the engine reads both.
    enum class SurfaceBuffer : uint8_t {
        DrawingBuffer,
        DisplayBuffer
    };

    // Draws the source buffer to the canvasBase().buffer().
    virtual RefPtr<ImageBuffer> surfaceBufferToImageBuffer(SurfaceBuffer);
    virtual bool isSurfaceBufferTransparentBlack(SurfaceBuffer) const;
    bool delegatesDisplay() const;
    virtual RefPtr<GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate();
    virtual void setContentsToLayer(GraphicsLayer&);

    // Returns the drawing buffer and runs the compositing steps of transferToImageBitmap.
    virtual RefPtr<ImageBuffer> transferToImageBuffer();

    bool hasActiveInspectorCanvasCallTracer() const { return m_hasActiveInspectorCanvasCallTracer; }
    void setHasActiveInspectorCanvasCallTracer(bool hasActiveInspectorCanvasCallTracer) { m_hasActiveInspectorCanvasCallTracer = hasActiveInspectorCanvasCallTracer; }

    // Returns true if there are pending deferred operations that might consume memory.
    virtual bool hasDeferredOperations() const { return false; }

    // Called periodically if needsFlush() was true when canvas change happened.
    virtual void flushDeferredOperations() { }

    virtual bool compositingResultsNeedUpdating() const { return false; }
    virtual bool needsPreparationForDisplay() const { return false; }
    // Swaps the current drawing buffer to display buffer.
    virtual void prepareForDisplay() { }

    virtual ImageBufferPixelFormat pixelFormat() const;
    virtual DestinationColorSpace colorSpace() const;
    virtual bool willReadFrequently() const;
    virtual std::optional<RenderingMode> renderingModeForTesting() const { return std::nullopt; }

    void setIsInPreparationForDisplayOrFlush(bool flag) { m_isInPreparationForDisplayOrFlush = flag; }
    bool isInPreparationForDisplayOrFlush() const { return m_isInPreparationForDisplayOrFlush; }

protected:
    enum class Type : uint8_t {
        CanvasElement2D,
        Offscreen2D,
        Paint,
        BitmapRenderer,
        Placeholder,
        WebGL1,
        WebGL2,
        WebGPU,
    };

    explicit CanvasRenderingContext(CanvasBase&, Type);
    bool taintsOrigin(const CanvasPattern*);
    bool taintsOrigin(const CanvasBase*);
    bool taintsOrigin(const CachedImage*);
    bool taintsOrigin(const HTMLImageElement*);
    bool taintsOrigin(const SVGImageElement*);
    bool taintsOrigin(const HTMLVideoElement*);
    bool taintsOrigin(const ImageBitmap*);
    bool taintsOrigin(const URL&);

    template<class T> void checkOrigin(const T* arg)
    {
        if (m_canvas->originClean() && taintsOrigin(arg))
            m_canvas->setOriginTainted();
    }
    void checkOrigin(const URL&);
    void checkOrigin(const CSSStyleImageValue&);

    bool m_isInPreparationForDisplayOrFlush { false };
    bool m_hasActiveInspectorCanvasCallTracer { false };

private:
    static Lock s_instancesLock;

    WeakRef<CanvasBase> m_canvas;
    const Type m_type;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CANVASRENDERINGCONTEXT(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToValueTypeName) \
    static bool isType(const WebCore::CanvasRenderingContext& context) { return context.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
