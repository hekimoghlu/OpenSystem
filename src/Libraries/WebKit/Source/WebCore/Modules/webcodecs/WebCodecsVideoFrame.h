/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

#if ENABLE(WEB_CODECS)

#include "ContextDestructionObserver.h"
#include "DOMRectReadOnly.h"
#include "JSDOMPromiseDeferredForward.h"
#include "PlaneLayout.h"
#include "VideoColorSpaceInit.h"
#include "WebCodecsAlphaOption.h"
#include "WebCodecsVideoFrameData.h"

namespace WebCore {

class BufferSource;
class CSSStyleImageValue;
class DOMRectReadOnly;
class HTMLCanvasElement;
class HTMLImageElement;
class HTMLVideoElement;
class ImageBitmap;
class ImageBuffer;
class NativeImage;
class OffscreenCanvas;
class SVGImageElement;
class VideoColorSpace;

template<typename> class ExceptionOr;

class WebCodecsVideoFrame : public RefCounted<WebCodecsVideoFrame>, public ContextDestructionObserver {
public:
    ~WebCodecsVideoFrame();

    using CanvasImageSource = std::variant<RefPtr<HTMLImageElement>
        , RefPtr<SVGImageElement>
        , RefPtr<HTMLCanvasElement>
        , RefPtr<ImageBitmap>
        , RefPtr<CSSStyleImageValue>
#if ENABLE(OFFSCREEN_CANVAS)
        , RefPtr<OffscreenCanvas>
#endif
#if ENABLE(VIDEO)
        , RefPtr<HTMLVideoElement>
#endif
    >;

    enum class AlphaOption { Keep, Discard };
    struct Init {
        std::optional<uint64_t> duration;
        std::optional<int64_t> timestamp;
        WebCodecsAlphaOption alpha { WebCodecsAlphaOption::Keep };

        std::optional<DOMRectInit> visibleRect;

        std::optional<size_t> displayWidth;
        std::optional<size_t> displayHeight;
    };
    struct BufferInit {
        VideoPixelFormat format { VideoPixelFormat::I420 };
        size_t codedWidth { 0 };
        size_t codedHeight { 0 };
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration { };

        std::optional<Vector<PlaneLayout>> layout { };

        std::optional<DOMRectInit> visibleRect { };

        std::optional<size_t> displayWidth { };
        std::optional<size_t> displayHeight { };

        std::optional<VideoColorSpaceInit> colorSpace { };
    };

    static ExceptionOr<Ref<WebCodecsVideoFrame>> create(ScriptExecutionContext&, CanvasImageSource&&, Init&&);
    static ExceptionOr<Ref<WebCodecsVideoFrame>> create(ScriptExecutionContext&, Ref<WebCodecsVideoFrame>&&, Init&&);
    static ExceptionOr<Ref<WebCodecsVideoFrame>> create(ScriptExecutionContext&, BufferSource&&, BufferInit&&);
    static ExceptionOr<Ref<WebCodecsVideoFrame>> create(ScriptExecutionContext&, ImageBuffer&, IntSize, Init&&);
    static Ref<WebCodecsVideoFrame> create(ScriptExecutionContext&, Ref<VideoFrame>&&, BufferInit&&);
    static Ref<WebCodecsVideoFrame> create(ScriptExecutionContext& context, WebCodecsVideoFrameData&& data) { return adoptRef(*new WebCodecsVideoFrame(context, WTFMove(data))); }

    std::optional<VideoPixelFormat> format() const { return m_data.format; }
    size_t codedWidth() const { return m_data.codedWidth; }
    size_t codedHeight() const { return m_data.codedHeight; }

    DOMRectReadOnly* codedRect() const;
    DOMRectReadOnly* visibleRect() const;

    size_t displayWidth() const { return m_data.displayWidth; }
    size_t displayHeight() const { return m_data.displayHeight; }
    std::optional<uint64_t> duration() const { return m_data.duration; }
    int64_t timestamp() const { return m_data.timestamp; }
    VideoColorSpace& colorSpace() const;

    struct CopyToOptions {
        std::optional<DOMRectInit> rect;
        std::optional<Vector<PlaneLayout>> layout;
    };
    ExceptionOr<size_t> allocationSize(const CopyToOptions&);

    using CopyToPromise = DOMPromiseDeferred<IDLSequence<IDLDictionary<PlaneLayout>>>;
    void copyTo(BufferSource&&, CopyToOptions&&, CopyToPromise&&);
    ExceptionOr<Ref<WebCodecsVideoFrame>> clone(ScriptExecutionContext&);
    void close();

    bool isDetached() const { return m_isDetached; }
    RefPtr<VideoFrame> internalFrame() const { return m_data.internalFrame; }

    void setDisplaySize(size_t, size_t);
    void setVisibleRect(const DOMRectInit&);
    bool shoudlDiscardAlpha() const { return m_data.format && (*m_data.format == VideoPixelFormat::RGBX || *m_data.format == VideoPixelFormat::BGRX); }

    const WebCodecsVideoFrameData& data() const { return m_data; }

    size_t memoryCost() const { return m_data.memoryCost(); }

private:
    explicit WebCodecsVideoFrame(ScriptExecutionContext&);
    WebCodecsVideoFrame(ScriptExecutionContext&, WebCodecsVideoFrameData&&);

    static ExceptionOr<Ref<WebCodecsVideoFrame>> initializeFrameFromOtherFrame(ScriptExecutionContext&, Ref<WebCodecsVideoFrame>&&, Init&&, VideoFrame::ShouldCloneWithDifferentTimestamp);
    static ExceptionOr<Ref<WebCodecsVideoFrame>> initializeFrameFromOtherFrame(ScriptExecutionContext&, Ref<VideoFrame>&&, Init&&, VideoFrame::ShouldCloneWithDifferentTimestamp);
    static ExceptionOr<Ref<WebCodecsVideoFrame>> initializeFrameWithResourceAndSize(ScriptExecutionContext&, Ref<NativeImage>&&, Init&&);

    WebCodecsVideoFrameData m_data;
    mutable RefPtr<VideoColorSpace> m_colorSpace;
    mutable RefPtr<DOMRectReadOnly> m_codedRect;
    mutable RefPtr<DOMRectReadOnly> m_visibleRect;
    bool m_isDetached { false };
};

}

#endif
