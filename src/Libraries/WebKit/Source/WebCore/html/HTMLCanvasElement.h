/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

#include "ActiveDOMObject.h"
#include "CanvasBase.h"
#include "Document.h"
#include "FloatRect.h"
#include "GraphicsTypes.h"
#include "HTMLElement.h"
#include <memory>
#include <wtf/Forward.h>

#if ENABLE(WEBGL)
#include "WebGLContextAttributes.h"
#endif

namespace WebCore {

class BlobCallback;
class CanvasRenderingContext;
class CanvasRenderingContext2D;
class GPU;
class GPUCanvasContext;
class GraphicsContext;
class Image;
class ImageBitmapRenderingContext;
class ImageBuffer;
class ImageData;
class MediaStream;
class OffscreenCanvas;
class VideoFrame;
class WebGLRenderingContextBase;
class WebCoreOpaqueRoot;
struct CanvasRenderingContext2DSettings;
struct ImageBitmapRenderingContextSettings;
struct UncachedString;

class HTMLCanvasElement final : public HTMLElement, public CanvasBase, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLCanvasElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLCanvasElement);
public:
    USING_CAN_MAKE_WEAKPTR(HTMLElement);

    static Ref<HTMLCanvasElement> create(Document&);
    static Ref<HTMLCanvasElement> create(const QualifiedName&, Document&);
    virtual ~HTMLCanvasElement();

    WEBCORE_EXPORT ExceptionOr<void> setWidth(unsigned);
    WEBCORE_EXPORT ExceptionOr<void> setHeight(unsigned);

    void setSize(const IntSize& newSize) override;

    CanvasRenderingContext* renderingContext() const final { return m_context.get(); }
    ExceptionOr<std::optional<RenderingContext>> getContext(JSC::JSGlobalObject&, const String& contextId, FixedVector<JSC::Strong<JSC::Unknown>>&& arguments);

    CanvasRenderingContext* getContext(const String&);

    static bool is2dType(const String&);
    CanvasRenderingContext2D* createContext2d(const String&, CanvasRenderingContext2DSettings&&);
    CanvasRenderingContext2D* getContext2d(const String&, CanvasRenderingContext2DSettings&&);

#if ENABLE(WEBGL)
    static bool isWebGLType(const String&);
    static WebGLVersion toWebGLVersion(const String&);
    WebGLRenderingContextBase* createContextWebGL(WebGLVersion type, WebGLContextAttributes&& = { });
    WebGLRenderingContextBase* getContextWebGL(WebGLVersion type, WebGLContextAttributes&& = { });
#endif

    static bool isBitmapRendererType(const String&);
    ImageBitmapRenderingContext* createContextBitmapRenderer(const String&, ImageBitmapRenderingContextSettings&&);
    ImageBitmapRenderingContext* getContextBitmapRenderer(const String&, ImageBitmapRenderingContextSettings&&);

    static bool isWebGPUType(const String&);
    GPUCanvasContext* createContextWebGPU(const String&, GPU*);
    GPUCanvasContext* getContextWebGPU(const String&, GPU*);

    WEBCORE_EXPORT ExceptionOr<UncachedString> toDataURL(const String& mimeType, JSC::JSValue quality);
    WEBCORE_EXPORT ExceptionOr<UncachedString> toDataURL(const String& mimeType);
    ExceptionOr<void> toBlob(Ref<BlobCallback>&&, const String& mimeType, JSC::JSValue quality);
#if ENABLE(OFFSCREEN_CANVAS)
    ExceptionOr<Ref<OffscreenCanvas>> transferControlToOffscreen();
#endif

    // Used for rendering
    void didDraw(const std::optional<FloatRect>&, ShouldApplyPostProcessingToDirtyRect) final;

    void paint(GraphicsContext&, const LayoutRect&);

#if ENABLE(MEDIA_STREAM) || ENABLE(WEB_CODECS)
    // Returns the context drawing buffer as a VideoFrame.
    RefPtr<VideoFrame> toVideoFrame();
#endif
#if ENABLE(MEDIA_STREAM)
    ExceptionOr<Ref<MediaStream>> captureStream(std::optional<double>&& frameRequestRate);
#endif

    const CSSParserContext& cssParserContext() const final;

    Image* copiedImage() const final;
    void clearCopiedImage() const final;
    RefPtr<ImageData> getImageData();

    SecurityOrigin* securityOrigin() const final;

    // FIXME(https://bugs.webkit.org/show_bug.cgi?id=275100): Only some canvas rendering contexts need an ImageBuffer.
    // It would be better to have the contexts own the buffers.
    void setImageBufferAndMarkDirty(RefPtr<ImageBuffer>&&) final;

    bool needsPreparationForDisplay();
    void prepareForDisplay();

    void setIsSnapshotting(bool isSnapshotting) { m_isSnapshotting = isSnapshotting; }
    bool isSnapshotting() const { return m_isSnapshotting; }

    bool isControlledByOffscreen() const;

    void queueTaskKeepingObjectAlive(TaskSource, Function<void()>&&) final;
    void dispatchEvent(Event&) final;

    // ActiveDOMObject.
    void ref() const final { HTMLElement::ref(); }
    void deref() const final { HTMLElement::deref(); }

    using HTMLElement::scriptExecutionContext;

private:
    HTMLCanvasElement(const QualifiedName&, Document&);

    bool isHTMLCanvasElement() const final { return true; }

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    // EventTarget.
    void eventListenersDidChange() final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool isReplaced(const RenderStyle&) const final;

    bool canContainRangeEndPoint() const final;
    bool canStartSelection() const final;

    void reset();

    void createImageBuffer() const final;
    void clearImageBuffer() const;

    void setSurfaceSize(const IntSize&);

    bool usesContentsAsLayerContents() const;

    ScriptExecutionContext* canvasBaseScriptExecutionContext() const final { return HTMLElement::scriptExecutionContext(); }

    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    std::unique_ptr<CanvasRenderingContext> m_context;
    mutable RefPtr<Image> m_copiedImage; // FIXME: This is temporary for platforms that have to copy the image buffer to render (and for CSSCanvasValue).
    mutable std::unique_ptr<CSSParserContext> m_cssParserContext;
    bool m_ignoreReset { false };
    mutable bool m_didClearImageBuffer { false };
#if ENABLE(WEBGL)
    bool m_hasRelevantWebGLEventListener { false };
#endif
    bool m_isSnapshotting { false };
};

WebCoreOpaqueRoot root(HTMLCanvasElement*);

} // namespace WebCore

namespace WTF {
template<typename ArgType> class TypeCastTraits<const WebCore::HTMLCanvasElement, ArgType, false /* isBaseType */> {
public:
    static bool isOfType(ArgType& node) { return checkTagName(node); }
private:
    static bool checkTagName(const WebCore::CanvasBase& base) { return base.isHTMLCanvasElement(); }
    static bool checkTagName(const WebCore::HTMLElement& element) { return element.hasTagName(WebCore::HTMLNames::canvasTag); }
    static bool checkTagName(const WebCore::Node& node) { return node.hasTagName(WebCore::HTMLNames::canvasTag); }
    static bool checkTagName(const WebCore::EventTarget& target)
    {
        auto* node = dynamicDowncast<WebCore::Node>(target);
        return node && checkTagName(*node);
    }
};
}
