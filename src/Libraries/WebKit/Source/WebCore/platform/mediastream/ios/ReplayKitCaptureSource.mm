/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#include "ReplayKitCaptureSource.h"

#if ENABLE(MEDIA_STREAM) && HAVE(REPLAYKIT)

#import "Logging.h"
#import "RealtimeVideoUtilities.h"
#import <wtf/BlockPtr.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/UUID.h>
#import <wtf/text/StringToIntegerConversion.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/ios/ReplayKitSoftLink.h>

using namespace WebCore;
@interface WebCoreReplayKitScreenRecorderHelper : NSObject {
    WeakPtr<ReplayKitCaptureSource> _capturer;
}

- (instancetype)initWithCallback:(WeakPtr<ReplayKitCaptureSource>&&)capturer;
- (void)disconnect;
- (void)observeValueForKeyPath:keyPath ofObject:(id)object change:(NSDictionary*)change context:(void*)context;
@end

@implementation WebCoreReplayKitScreenRecorderHelper
- (instancetype)initWithCallback:(WeakPtr<ReplayKitCaptureSource>&&)capturer
{
    self = [super init];
    if (!self)
        return self;

    _capturer = WTFMove(capturer);
    [[PAL::getRPScreenRecorderClass() sharedRecorder] addObserver:self forKeyPath:@"recording" options:NSKeyValueObservingOptionNew context:(void *)nil];

    return self;
}

- (void)disconnect
{
    _capturer = nullptr;
    [[PAL::getRPScreenRecorderClass() sharedRecorder] removeObserver:self forKeyPath:@"recording"];
}

- (void)observeValueForKeyPath:keyPath ofObject:(id)object change:(NSDictionary*)change context:(void*)context
{
    UNUSED_PARAM(object);
    UNUSED_PARAM(context);

    RefPtr protectedCapturer = _capturer.get();
    if (!protectedCapturer)
        return;

    id newValue = [change valueForKey:NSKeyValueChangeNewKey];
    bool willChange = [[change valueForKey:NSKeyValueChangeNotificationIsPriorKey] boolValue];

#if !RELEASE_LOG_DISABLED
    if (protectedCapturer->loggerPtr()) {
        auto identifier = Logger::LogSiteIdentifier("ReplayKitCaptureSource"_s, "observeValueForKeyPath"_s, protectedCapturer->logIdentifier());
        RetainPtr<NSString> valueString = adoptNS([[NSString alloc] initWithFormat:@"%@", newValue]);
        protectedCapturer->logger().logAlways(protectedCapturer->logChannel(), identifier, willChange ? "will" : "did", " change '", [keyPath UTF8String], "' to ", [valueString.get() UTF8String]);
    }
#endif

    if (willChange)
        return;

    if ([keyPath isEqualToString:@"recording"])
        protectedCapturer->captureStateDidChange();
}
@end

namespace WebCore {

bool ReplayKitCaptureSource::isAvailable()
{
    return [PAL::getRPScreenRecorderClass() sharedRecorder].isAvailable;
}

ReplayKitCaptureSource::ReplayKitCaptureSource(CapturerObserver& observer)
    : DisplayCaptureSourceCocoa::Capturer(observer)
    , m_captureWatchdogTimer(*this, &ReplayKitCaptureSource::verifyCaptureIsActive)
{
}

ReplayKitCaptureSource::~ReplayKitCaptureSource()
{
    [m_recorderHelper disconnect];
    stop();
    m_currentFrame = nullptr;
}

bool ReplayKitCaptureSource::start()
{
    ASSERT(isAvailable());

    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG_IF(loggerPtr(), identifier);

    auto *screenRecorder = [PAL::getRPScreenRecorderClass() sharedRecorder];
    if (screenRecorder.recording)
        return true;

#if !PLATFORM(APPLETV)
    // FIXME: Add support for concurrent audio capture.
    [screenRecorder setMicrophoneEnabled:NO];
#endif

    if (!m_recorderHelper)
        m_recorderHelper = ([[WebCoreReplayKitScreenRecorderHelper alloc] initWithCallback:this]);

    auto captureHandler = makeBlockPtr([this, weakThis = WeakPtr { *this }, identifier](CMSampleBufferRef _Nonnull sampleBuffer, RPSampleBufferType bufferType, NSError * _Nullable error) {

        if (bufferType != RPSampleBufferTypeVideo)
            return;

        ERROR_LOG_IF(error && loggerPtr(), identifier, "startCaptureWithHandler failed ", error);

        ++m_frameCount;

        RunLoop::main().dispatch([weakThis, sampleBuffer = retainPtr(sampleBuffer)]() mutable {
            if (!weakThis)
                return;

            weakThis->screenRecorderDidOutputVideoSample(WTFMove(sampleBuffer));
        });
    });

    auto completionHandler = makeBlockPtr([this, weakThis = WeakPtr { *this }, identifier](NSError * _Nullable error) {
        // FIXME: It should be safe to call `videoFrameAvailable` from any thread. Test this and get rid of this main thread hop.
        RunLoop::main().dispatch([this, weakThis, error = retainPtr(error), identifier]() mutable {
            if (!weakThis || !error)
                return;

            ERROR_LOG_IF(loggerPtr(), identifier, "completionHandler failed ", error.get());
            weakThis->stop();
        });
    });

    [screenRecorder startCaptureWithHandler:captureHandler.get() completionHandler:completionHandler.get()];

    return true;
}

void ReplayKitCaptureSource::screenRecorderDidOutputVideoSample(RetainPtr<CMSampleBufferRef>&& sampleBuffer)
{
    m_currentFrame = sampleBuffer.get();
    m_intrinsicSize = IntSize(PAL::CMVideoFormatDescriptionGetPresentationDimensions(PAL::CMSampleBufferGetFormatDescription(m_currentFrame.get()), true, true));
}

void ReplayKitCaptureSource::captureStateDidChange()
{
    RunLoop::main().dispatch([this, weakThis = WeakPtr { *this }, identifier = LOGIDENTIFIER]() mutable {
        if (!weakThis)
            return;

        bool isRecording = !![[PAL::getRPScreenRecorderClass() sharedRecorder] isRecording];
        if (m_isRunning == (isRecording && !m_interrupted))
            return;

        m_isRunning = isRecording && !m_interrupted;
        ALWAYS_LOG_IF(loggerPtr(), identifier, m_isRunning);
        isRunningChanged(m_isRunning);
    });
}

void ReplayKitCaptureSource::stop()
{
    auto identifier = LOGIDENTIFIER;
    ALWAYS_LOG_IF(loggerPtr(), identifier);
    m_captureWatchdogTimer.stop();
    m_interrupted = false;
    m_isRunning = false;

    auto *screenRecorder = [PAL::getRPScreenRecorderClass() sharedRecorder];
    if (screenRecorder.recording) {
        [screenRecorder stopCaptureWithHandler:^(NSError * _Nullable error) {
            ERROR_LOG_IF(error && loggerPtr(), identifier, "startCaptureWithHandler failed ", error);
        }];
    }
}

DisplayCaptureSourceCocoa::DisplayFrameType ReplayKitCaptureSource::generateFrame()
{
    return m_currentFrame;
}

void ReplayKitCaptureSource::verifyCaptureIsActive()
{
    ASSERT(m_isRunning || m_interrupted);
    auto identifier = LOGIDENTIFIER;
    if (m_lastFrameCount != m_frameCount) {
        m_lastFrameCount = m_frameCount;
        if (m_interrupted) {
            ALWAYS_LOG_IF(loggerPtr(), identifier, "frame received after interruption, unmuting");
            m_interrupted = false;
            captureStateDidChange();
        }
        return;
    }

    ALWAYS_LOG_IF(loggerPtr(), identifier, "no frame received in ", static_cast<int>(m_captureWatchdogTimer.repeatInterval().value()), " seconds, muting");
    m_interrupted = true;
    captureStateDidChange();
}

void ReplayKitCaptureSource::startCaptureWatchdogTimer()
{
    static constexpr Seconds verifyCaptureInterval = 2_s;
    if (m_captureWatchdogTimer.isActive())
        return;

    m_captureWatchdogTimer.startRepeating(verifyCaptureInterval);
    m_lastFrameCount = m_frameCount;
}

static String screenDeviceUUID()
{
    static NeverDestroyed<String> screenID = createVersion4UUIDString();
    return screenID;
}

static CaptureDevice& screenDevice()
{
    static NeverDestroyed<CaptureDevice> device = { screenDeviceUUID(), CaptureDevice::DeviceType::Screen, "Screen 1"_str, emptyString(), true };
    return device;
}

std::optional<CaptureDevice> ReplayKitCaptureSource::screenCaptureDeviceWithPersistentID(const String& displayID)
{
    if (!isAvailable()) {
        RELEASE_LOG_ERROR(WebRTC, "ReplayKitCaptureSource::screenCaptureDeviceWithPersistentID: screen capture unavailable");
        return std::nullopt;
    }

    if (displayID != screenDeviceUUID()) {
        RELEASE_LOG_ERROR(WebRTC, "ReplayKitCaptureSource::screenCaptureDeviceWithPersistentID: invalid display ID");
        return std::nullopt;
    }

    return screenDevice();
}

void ReplayKitCaptureSource::screenCaptureDevices(Vector<CaptureDevice>& displays)
{
    if (isAvailable())
        displays.append(screenDevice());
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && HAVE(REPLAYKIT)
