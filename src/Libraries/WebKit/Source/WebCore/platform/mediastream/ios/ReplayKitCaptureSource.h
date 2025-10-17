/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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

#if ENABLE(MEDIA_STREAM) && HAVE(REPLAYKIT)

#include "DisplayCaptureSourceCocoa.h"
#include "Timer.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS RPScreenRecorder;
OBJC_CLASS WebCoreReplayKitScreenRecorderHelper;

namespace WebCore {

class ReplayKitCaptureSource final : public DisplayCaptureSourceCocoa::Capturer, public CanMakeWeakPtr<ReplayKitCaptureSource> {
public:
    explicit ReplayKitCaptureSource(CapturerObserver&);
    virtual ~ReplayKitCaptureSource();

    static bool isAvailable();

    static std::optional<CaptureDevice> screenCaptureDeviceWithPersistentID(const String&);
    static void screenCaptureDevices(Vector<CaptureDevice>&);

    void captureStateDidChange();

private:

    // DisplayCaptureSourceCocoa::Capturer
    bool start() final;
    void stop() final;
    DisplayCaptureSourceCocoa::DisplayFrameType generateFrame() final;
    CaptureDevice::DeviceType deviceType() const final { return CaptureDevice::DeviceType::Screen; }
    DisplaySurfaceType surfaceType() const final { return DisplaySurfaceType::Monitor; }
    virtual void commitConfiguration(const RealtimeMediaSourceSettings&) { }
    virtual IntSize intrinsicSize() const { return m_intrinsicSize; }

    // LoggerHelper
    ASCIILiteral logClassName() const final { return "ReplayKitCaptureSource"_s; }

    void screenRecorderDidOutputVideoSample(RetainPtr<CMSampleBufferRef>&&);
    void startCaptureWatchdogTimer();
    void verifyCaptureIsActive();

    RetainPtr<CMSampleBufferRef> m_currentFrame;
    RetainPtr<RPScreenRecorder> m_screenRecorder;
    RetainPtr<WebCoreReplayKitScreenRecorderHelper> m_recorderHelper;

    Timer m_captureWatchdogTimer;
    uint64_t m_frameCount { 0 };
    uint64_t m_lastFrameCount { 0 };
    IntSize m_intrinsicSize;
    bool m_isRunning { false };
    bool m_interrupted { false };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && HAVE(REPLAYKIT)
