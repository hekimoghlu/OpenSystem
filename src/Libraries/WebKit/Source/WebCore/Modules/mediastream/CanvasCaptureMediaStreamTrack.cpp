/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#include "CanvasCaptureMediaStreamTrack.h"

#if ENABLE(MEDIA_STREAM)

#include "GraphicsContext.h"
#include "HTMLCanvasElement.h"
#include "VideoFrame.h"
#include "WebGLRenderingContextBase.h"
#include <wtf/TZoneMallocInlines.h>

#if USE(GSTREAMER)
#include "VideoFrameGStreamer.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CanvasCaptureMediaStreamTrack);

Ref<CanvasCaptureMediaStreamTrack> CanvasCaptureMediaStreamTrack::create(Document& document, Ref<HTMLCanvasElement>&& canvas, std::optional<double>&& frameRequestRate)
{
    auto source = CanvasCaptureMediaStreamTrack::Source::create(canvas.get(), WTFMove(frameRequestRate));
    auto track = adoptRef(*new CanvasCaptureMediaStreamTrack(document, WTFMove(canvas), WTFMove(source)));
    track->suspendIfNeeded();
    return track;
}

CanvasCaptureMediaStreamTrack::CanvasCaptureMediaStreamTrack(Document& document, Ref<HTMLCanvasElement>&& canvas, Ref<CanvasCaptureMediaStreamTrack::Source>&& source)
    : MediaStreamTrack(document, MediaStreamTrackPrivate::create(document.logger(), source.copyRef()))
    , m_canvas(WTFMove(canvas))
{
}

CanvasCaptureMediaStreamTrack::CanvasCaptureMediaStreamTrack(Document& document, Ref<HTMLCanvasElement>&& canvas, Ref<MediaStreamTrackPrivate>&& privateTrack)
    : MediaStreamTrack(document, WTFMove(privateTrack))
    , m_canvas(WTFMove(canvas))
{
}

Ref<CanvasCaptureMediaStreamTrack::Source> CanvasCaptureMediaStreamTrack::Source::create(HTMLCanvasElement& canvas, std::optional<double>&& frameRequestRate)
{
    auto source = adoptRef(*new Source(canvas, WTFMove(frameRequestRate)));
    source->start();

    callOnMainThread([source] {
        if (!source->m_canvas)
            return;
        source->captureCanvas();
    });
    return source;
}

// FIXME: Give source id and name
CanvasCaptureMediaStreamTrack::Source::Source(HTMLCanvasElement& canvas, std::optional<double>&& frameRequestRate)
    : RealtimeMediaSource(CaptureDevice { { }, CaptureDevice::DeviceType::Camera, "CanvasCaptureMediaStreamTrack"_s })
    , m_frameRequestRate(WTFMove(frameRequestRate))
    , m_requestFrameTimer(*this, &Source::requestFrameTimerFired)
    , m_captureCanvasTimer(*this, &Source::captureCanvas)
    , m_canvas(&canvas)
{
}

void CanvasCaptureMediaStreamTrack::Source::startProducingData()
{
    if (!m_canvas)
        return;
    m_canvas->addObserver(*this);
    m_canvas->addDisplayBufferObserver(*this);

    if (!m_frameRequestRate)
        return;

    if (m_frameRequestRate.value())
        m_requestFrameTimer.startRepeating(1_s / m_frameRequestRate.value());
}

void CanvasCaptureMediaStreamTrack::Source::stopProducingData()
{
    m_requestFrameTimer.stop();

    if (!m_canvas)
        return;
    m_canvas->removeObserver(*this);
    m_canvas->removeDisplayBufferObserver(*this);
}

void CanvasCaptureMediaStreamTrack::Source::requestFrameTimerFired()
{
    requestFrame();
}

void CanvasCaptureMediaStreamTrack::Source::canvasDestroyed(CanvasBase& canvas)
{
    ASSERT_UNUSED(canvas, m_canvas == &canvas);

    stop();
    m_canvas = nullptr;
}

const RealtimeMediaSourceSettings& CanvasCaptureMediaStreamTrack::Source::settings()
{
    if (m_currentSettings)
        return m_currentSettings.value();

    RealtimeMediaSourceSupportedConstraints constraints;
    constraints.setSupportsWidth(true);
    constraints.setSupportsHeight(true);

    RealtimeMediaSourceSettings settings;
    settings.setWidth(m_canvas->width());
    settings.setHeight(m_canvas->height());
    settings.setSupportedConstraints(constraints);

    m_currentSettings = WTFMove(settings);
    return m_currentSettings.value();
}

void CanvasCaptureMediaStreamTrack::Source::settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag> settings)
{
    if (settings.containsAny({ RealtimeMediaSourceSettings::Flag::Width, RealtimeMediaSourceSettings::Flag::Height }))
        m_currentSettings = std::nullopt;
}

void CanvasCaptureMediaStreamTrack::Source::canvasResized(CanvasBase& canvas)
{
    ASSERT_UNUSED(canvas, m_canvas == &canvas);
    setSize(IntSize(m_canvas->width(), m_canvas->height()));
}

void CanvasCaptureMediaStreamTrack::Source::canvasChanged(CanvasBase&, const FloatRect&)
{
    // If canvas needs preparation, the capture will be scheduled once document prepares the canvas.
    if (m_canvas->needsPreparationForDisplay())
        return;
    scheduleCaptureCanvas();
}

void CanvasCaptureMediaStreamTrack::Source::scheduleCaptureCanvas()
{
    // FIXME: We should try to generate the frame at the time the screen is being updated.
    if (m_captureCanvasTimer.isActive())
        return;
    m_captureCanvasTimer.startOneShot(0_s);
}

void CanvasCaptureMediaStreamTrack::Source::canvasDisplayBufferPrepared(CanvasBase& canvas)
{
    ASSERT_UNUSED(canvas, m_canvas == &canvas);
    // FIXME: Here we should capture the image instead.
    // However, submitting the sample to the receiver might cause layout,
    // and currently the display preparation is done after layout.
    scheduleCaptureCanvas();
}

void CanvasCaptureMediaStreamTrack::Source::captureCanvas()
{
    ASSERT(m_canvas);
    Ref canvas = *m_canvas;

    if (!isProducingData())
        return;

    if (m_frameRequestRate) {
        if (!m_shouldEmitFrame)
            return;
        m_shouldEmitFrame = false;
    }

    if (!canvas->originClean())
        return;
    RefPtr<VideoFrame> videoFrame = [&]() -> RefPtr<VideoFrame> {
#if ENABLE(WEBGL)
        if (auto* gl = dynamicDowncast<WebGLRenderingContextBase>(canvas->renderingContext()))
            return gl->surfaceBufferToVideoFrame(CanvasRenderingContext::SurfaceBuffer::DisplayBuffer);
#endif
        return canvas->toVideoFrame();
    }();
    if (!videoFrame)
        return;

#if USE(GSTREAMER)
    auto gstVideoFrame = downcast<VideoFrameGStreamer>(videoFrame);
    if (m_frameRequestRate)
        gstVideoFrame->setFrameRate(*m_frameRequestRate);
    else {
        static const double s_frameRate = 60;
        gstVideoFrame->setMaxFrameRate(s_frameRate);
        gstVideoFrame->setPresentationTime(m_presentationTimeStamp);
        m_presentationTimeStamp = m_presentationTimeStamp + MediaTime::createWithDouble(1.0 / s_frameRate);
    }
#endif

    VideoFrameTimeMetadata metadata;
    metadata.captureTime = MonotonicTime::now().secondsSinceEpoch();
    videoFrameAvailable(*videoFrame, metadata);
}

RefPtr<MediaStreamTrack> CanvasCaptureMediaStreamTrack::clone()
{
    if (!scriptExecutionContext())
        return nullptr;
    
    auto track = adoptRef(*new CanvasCaptureMediaStreamTrack(downcast<Document>(*scriptExecutionContext()), m_canvas.copyRef(), m_private->clone()));
    track->suspendIfNeeded();
    return track;
}

}

#endif // ENABLE(MEDIA_STREAM)
