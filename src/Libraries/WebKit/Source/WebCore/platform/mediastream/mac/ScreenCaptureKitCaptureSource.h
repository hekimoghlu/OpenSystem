/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#if HAVE(SCREEN_CAPTURE_KIT)

#include "DisplayCaptureManager.h"
#include "DisplayCaptureSourceCocoa.h"
#include "ScreenCaptureKitSharingSessionManager.h"
#include <wtf/BlockPtr.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS NSDictionary;
OBJC_CLASS NSError;
OBJC_CLASS SCDisplay;
OBJC_CLASS SCShareableContent;
OBJC_CLASS SCStream;
OBJC_CLASS SCContentFilter;
OBJC_CLASS SCContentSharingSession;
OBJC_CLASS SCStreamConfiguration;
OBJC_CLASS SCWindow;
OBJC_CLASS WebCoreScreenCaptureKitHelper;
using CMSampleBufferRef = struct opaqueCMSampleBuffer*;

namespace WebCore {

class ImageTransferSessionVT;

class ScreenCaptureKitCaptureSource final
    : public DisplayCaptureSourceCocoa::Capturer
    , public ScreenCaptureSessionSourceObserver {
public:
    static Expected<uint32_t, CaptureSourceError> computeDeviceID(const CaptureDevice&);

    ScreenCaptureKitCaptureSource(CapturerObserver&, const CaptureDevice&, uint32_t);
    virtual ~ScreenCaptureKitCaptureSource();

    WEBCORE_EXPORT static bool isAvailable();

    static std::optional<CaptureDevice> screenCaptureDeviceWithPersistentID(const String&);
    static std::optional<CaptureDevice> windowCaptureDeviceWithPersistentID(const String&);

    using Content = std::variant<RetainPtr<SCWindow>, RetainPtr<SCDisplay>>;
    void streamDidOutputVideoSampleBuffer(RetainPtr<CMSampleBufferRef>);
    void sessionFailedWithError(RetainPtr<NSError>&&, const String&);
    void outputVideoEffectDidStartForStream() { m_isVideoEffectEnabled = true; }
    void outputVideoEffectDidStopForStream() { m_isVideoEffectEnabled = false; }

private:
    // DisplayCaptureSourceCocoa::Capturer
    bool start() final;
    void stop() final { stopInternal([] { }); }
    void end() final;
    DisplayCaptureSourceCocoa::DisplayFrameType generateFrame() final;
    CaptureDevice::DeviceType deviceType() const final;
    DisplaySurfaceType surfaceType() const final;
    void commitConfiguration(const RealtimeMediaSourceSettings&) final;
    IntSize intrinsicSize() const final;
    void whenReady(CompletionHandler<void(CaptureSourceError&&)>&&) final;

    // LoggerHelper
    ASCIILiteral logClassName() const final { return "ScreenCaptureKitCaptureSource"_s; }

    // ScreenCaptureKitSharingSessionManager::Observer
    void sessionFilterDidChange(SCContentFilter*) final;
    void sessionStreamDidEnd(SCStream*) final;

    void stopInternal(CompletionHandler<void()>&&);
    void startContentStream();
    void findShareableContent();
    RetainPtr<SCStreamConfiguration> streamConfiguration();
    void updateStreamConfiguration();

    dispatch_queue_t captureQueue();

    SCStream* contentStream() const { return m_sessionSource ? m_sessionSource->stream() : nullptr; }
    SCContentFilter* contentFilter() const { return m_sessionSource ? m_sessionSource->contentFilter() : nullptr; }

    void clearSharingSession();

    std::optional<Content> m_content;
    RetainPtr<WebCoreScreenCaptureKitHelper> m_captureHelper;
    RetainPtr<SCContentSharingSession> m_sharingSession;
    RetainPtr<SCContentFilter> m_contentFilter;
    RetainPtr<CMSampleBufferRef> m_currentFrame;
    RefPtr<ScreenCaptureSessionSource> m_sessionSource;
    RetainPtr<SCStreamConfiguration> m_streamConfiguration;
    OSObjectPtr<dispatch_queue_t> m_captureQueue;
    CaptureDevice m_captureDevice;
    uint32_t m_deviceID { 0 };
    mutable std::optional<IntSize> m_intrinsicSize;
    std::unique_ptr<ImageTransferSessionVT> m_transferSession;

    FloatSize m_contentSize;
    uint32_t m_width { 0 };
    uint32_t m_height { 0 };
    float m_frameRate { 0 };
    bool m_isRunning { false };
    bool m_isVideoEffectEnabled { false };
    bool m_didReceiveVideoFrame { false };
    CompletionHandler<void(CaptureSourceError&&)> m_whenReadyCallback;
};

} // namespace WebCore

#endif // HAVE(SCREEN_CAPTURE_KIT)
