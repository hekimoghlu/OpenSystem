/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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

#if ENABLE(MEDIA_STREAM)

#include "CanvasBase.h"
#include "CanvasObserver.h"
#include "MediaStreamTrack.h"
#include "Timer.h"
#include <wtf/TypeCasts.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class HTMLCanvasElement;
class Image;

class CanvasCaptureMediaStreamTrack final : public MediaStreamTrack {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CanvasCaptureMediaStreamTrack);
public:
    static Ref<CanvasCaptureMediaStreamTrack> create(Document&, Ref<HTMLCanvasElement>&&, std::optional<double>&& frameRequestRate);

    HTMLCanvasElement& canvas() { return m_canvas.get(); }
    void requestFrame() { static_cast<Source&>(source()).requestFrame(); }

    RefPtr<MediaStreamTrack> clone() final;

private:
    class Source final : public RealtimeMediaSource, private CanvasObserver, private CanvasDisplayBufferObserver, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<Source, WTF::DestructionThread::MainRunLoop> {
    public:
        static Ref<Source> create(HTMLCanvasElement&, std::optional<double>&& frameRequestRate);
        
        void requestFrame() { m_shouldEmitFrame = true; }
        std::optional<double> frameRequestRate() const { return m_frameRequestRate; }

        WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

    private:
        Source(HTMLCanvasElement&, std::optional<double>&&);

        // CanvasObserver overrides.
        void canvasChanged(CanvasBase&, const FloatRect&) final;
        void canvasResized(CanvasBase&) final;
        void canvasDestroyed(CanvasBase&) final;

        // CanvasDisplayBufferObserver overrides.
        void canvasDisplayBufferPrepared(CanvasBase&) final;

        // RealtimeMediaSource overrides.
        void startProducingData() final;
        void stopProducingData()  final;
        const RealtimeMediaSourceCapabilities& capabilities() final { return RealtimeMediaSourceCapabilities::emptyCapabilities(); }
        const RealtimeMediaSourceSettings& settings() final;
        void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final;
        void scheduleCaptureCanvas();
        void captureCanvas();
        void requestFrameTimerFired();

        bool m_shouldEmitFrame { true };
        std::optional<double> m_frameRequestRate;
        Timer m_requestFrameTimer;
        Timer m_captureCanvasTimer;
        std::optional<RealtimeMediaSourceSettings> m_currentSettings;
        WeakPtr<HTMLCanvasElement, WeakPtrImplWithEventTargetData> m_canvas;
        RefPtr<Image> m_currentImage;
#if USE(GSTREAMER)
        MediaTime m_presentationTimeStamp { MediaTime::zeroTime() };
#endif
    };

    CanvasCaptureMediaStreamTrack(Document&, Ref<HTMLCanvasElement>&&, Ref<Source>&&);
    CanvasCaptureMediaStreamTrack(Document&, Ref<HTMLCanvasElement>&&, Ref<MediaStreamTrackPrivate>&&);

    bool isCanvas() const final { return true; }

    Ref<HTMLCanvasElement> m_canvas;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CanvasCaptureMediaStreamTrack)
static bool isType(const WebCore::MediaStreamTrack& track) { return track.isCanvas(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_STREAM)
