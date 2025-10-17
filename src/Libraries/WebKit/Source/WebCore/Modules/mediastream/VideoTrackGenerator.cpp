/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "VideoTrackGenerator.h"

#if ENABLE(MEDIA_STREAM) && ENABLE(WEB_CODECS)

#include "Exception.h"
#include "JSWebCodecsVideoFrame.h"
#include "MediaStreamTrack.h"
#include "VideoFrame.h"
#include "WritableStream.h"
#include "WritableStreamSink.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(VideoTrackGenerator);

ExceptionOr<Ref<VideoTrackGenerator>> VideoTrackGenerator::create(ScriptExecutionContext& context)
{
    auto source = Source::create(context.identifier());
    auto sink = Sink::create(Ref { source });
    auto writableOrException = WritableStream::create(*JSC::jsCast<JSDOMGlobalObject*>(context.globalObject()), Ref { sink });

    if (writableOrException.hasException())
        return writableOrException.releaseException();

    Ref writable = writableOrException.releaseReturnValue();
    source->setWritable(writable.get());

    // FIXME: Maybe we should have the writable buffer frames until the source is actually started.
    callOnMainThread([source] {
        source->start();
    });

    auto logger = Logger::create(&context);
    logger->setEnabled(&context, context.isAlwaysOnLoggingAllowed());

    auto privateTrack = MediaStreamTrackPrivate::create(WTFMove(logger), WTFMove(source), [identifier = context.identifier()](Function<void()>&& task) {
        ScriptExecutionContext::postTaskTo(identifier, [task = WTFMove(task)] (auto&) mutable {
            task();
        });
    });

    RealtimeMediaSourceSupportedConstraints supportedConstraints;
    supportedConstraints.setSupportsWidth(true);
    supportedConstraints.setSupportsHeight(true);

    RealtimeMediaSourceSettings settings;
    settings.setSupportedConstraints(supportedConstraints);
    settings.setWidth(0);
    settings.setHeight(0);

    RealtimeMediaSourceCapabilities capabilities { supportedConstraints };
    capabilities.setWidth({ 0, 0 });
    capabilities.setHeight({ 0, 0 });

    privateTrack->initializeSettings(WTFMove(settings));
    privateTrack->initializeCapabilities(WTFMove(capabilities));

    return adoptRef(*new VideoTrackGenerator(WTFMove(sink), WTFMove(writable), MediaStreamTrack::create(context, WTFMove(privateTrack))));
}

VideoTrackGenerator::VideoTrackGenerator(Ref<Sink>&& sink, Ref<WritableStream>&& writable, Ref<MediaStreamTrack>&& track)
    : m_sink(WTFMove(sink))
    , m_writable(WTFMove(writable))
    , m_track(WTFMove(track))
{
}

VideoTrackGenerator::~VideoTrackGenerator() = default;

void VideoTrackGenerator::setMuted(ScriptExecutionContext& context, bool muted)
{
    if (muted == m_muted)
        return;

    m_muted = muted;
    m_sink->setMuted(m_muted);

    if (m_hasMutedChanged)
        return;

    m_hasMutedChanged = true;
    context.postTask([this, protectedThis = Ref { *this }] (auto&) {
        m_hasMutedChanged = false;
        m_track->privateTrack().setMuted(m_muted);
    });
}

Ref<WritableStream> VideoTrackGenerator::writable()
{
    return Ref { m_writable };
}

Ref<MediaStreamTrack> VideoTrackGenerator::track()
{
    return Ref { m_track };
}

VideoTrackGenerator::Source::Source(ScriptExecutionContextIdentifier identifier)
    : RealtimeMediaSource(CaptureDevice { { }, CaptureDevice::DeviceType::Camera, emptyString() })
    , m_contextIdentifier(identifier)
{
}

void VideoTrackGenerator::Source::endProducingData()
{
    ASSERT(isMainThread());
    ScriptExecutionContext::postTaskTo(m_contextIdentifier, [weakThis = ThreadSafeWeakPtr { *this }] (auto&) {
        RefPtr protectedSource = weakThis.get();
        RefPtr writable = protectedSource ? protectedSource->m_writable.get() : nullptr;
        if (writable)
            writable->closeIfPossible();
    });
}

void VideoTrackGenerator::Source::setWritable(WritableStream& writable)
{
    ASSERT(!isMainThread());
    ASSERT(!m_writable);
    m_writable = writable;
}

void VideoTrackGenerator::Source::writeVideoFrame(VideoFrame& frame, VideoFrameTimeMetadata metadata)
{
    ASSERT(!isMainThread());

    auto frameSize = IntSize(frame.presentationSize());
    if (frame.rotation() == VideoFrame::Rotation::Left || frame.rotation() == VideoFrame::Rotation::Right)
        frameSize = frameSize.transposedSize();

    if (m_videoFrameSize != frameSize) {
        m_videoFrameSize = frameSize;
        callOnMainThread([this, protectedThis = Ref { *this }, frameSize] {
            RealtimeMediaSourceSupportedConstraints supportedConstraints;
            supportedConstraints.setSupportsWidth(true);
            supportedConstraints.setSupportsHeight(true);

            if (m_maxVideoFrameSize.width() < frameSize.width() || m_maxVideoFrameSize.height() < frameSize.height()) {
                m_maxVideoFrameSize.clampToMinimumSize(frameSize);

                m_capabilities = RealtimeMediaSourceCapabilities { supportedConstraints };
                m_capabilities.setWidth({ 0, m_maxVideoFrameSize.width() });
                m_capabilities.setHeight({ 0, m_maxVideoFrameSize.height() });
            }

            m_settings.setSupportedConstraints(supportedConstraints);
            m_settings.setWidth(frameSize.width());
            m_settings.setHeight(frameSize.height());

            setSize(frameSize);
        });
    }

    videoFrameAvailable(frame, metadata);
}

VideoTrackGenerator::Sink::Sink(Ref<Source>&& source)
    : m_source(WTFMove(source))
{
}

void VideoTrackGenerator::Sink::write(ScriptExecutionContext&, JSC::JSValue value, DOMPromiseDeferred<void>&& promise)
{
    auto* jsFrameObject = JSC::jsDynamicCast<JSWebCodecsVideoFrame*>(value);
    RefPtr frameObject = jsFrameObject ? &jsFrameObject->wrapped() : nullptr;
    if (!frameObject) {
        promise.reject(Exception { ExceptionCode::TypeError, "Expected a VideoFrame object"_s });
        return;
    }

    RefPtr videoFrame = frameObject->internalFrame();
    if (!videoFrame) {
        promise.reject(Exception { ExceptionCode::TypeError, "VideoFrame object is not valid"_s });
        return;
    }

    if (!m_muted)
        m_source->writeVideoFrame(*videoFrame, { });

    frameObject->close();
    promise.resolve();
}

void VideoTrackGenerator::Sink::close()
{
    callOnMainThread([source = m_source] {
        source->endImmediatly();
    });
}

void VideoTrackGenerator::Sink::error(String&&)
{
    close();
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && ENABLE(WEB_CODECS)
