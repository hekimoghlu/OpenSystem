/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#import "config.h"
#import "AVCaptureDeviceManager.h"

#if ENABLE(MEDIA_STREAM) && USE(AVFOUNDATION)

#import "AVVideoCaptureSource.h"
#import "AudioSourceProvider.h"
#import "Logging.h"
#import "MediaConstraints.h"
#import "RealtimeMediaSource.h"
#import "RealtimeMediaSourceCenter.h"
#import "RealtimeMediaSourceSettings.h"
#import "RealtimeMediaSourceSupportedConstraints.h"
#import <AVFoundation/AVCaptureDevice.h>
#import <AVFoundation/AVCaptureSession.h>
#import <objc/runtime.h>
#import <wtf/MainThread.h>
#import <wtf/NeverDestroyed.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

using namespace WebCore;

@interface WebCoreAVCaptureDeviceManagerObserver : NSObject
{
    AVCaptureDeviceManager* m_callback;
}

-(id)initWithCallback:(AVCaptureDeviceManager*)callback;
-(void)disconnect;
-(void)deviceConnectedDidChange:(NSNotification *)notification;
-(void)observeValueForKeyPath:keyPath ofObject:(id)object change:(NSDictionary*)change context:(void*)context;
@end

namespace WebCore {

void AVCaptureDeviceManager::computeCaptureDevices(CompletionHandler<void()>&& callback)
{
    if (!m_isInitialized) {
        refreshCaptureDevicesInternal([this, callback = WTFMove(callback)]() mutable {
            m_isInitialized = true;
            callback();
        }, ShouldSetUserPreferredCamera::Yes);
        return;
    }
    callback();
}

const Vector<CaptureDevice>& AVCaptureDeviceManager::captureDevices()
{
    ASSERT(m_isInitialized);
    RELEASE_LOG_ERROR_IF(!m_isInitialized, WebRTC, "Retrieving AVCaptureDeviceManager list before initialization");
    return m_devices;
}

inline static bool deviceIsAvailable(AVCaptureDevice *device)
{
#if HAVE(AVCAPTUREDEVICE)
    if (![device isConnected])
        return false;

#if !PLATFORM(IOS_FAMILY)
    if ([device isSuspended])
        return false;
#endif

    return true;
#else
    UNUSED_PARAM(device);
    return false;
#endif
}

RetainPtr<NSArray> AVCaptureDeviceManager::currentCameras()
{
#if HAVE(AVCAPTUREDEVICE)
    AVCaptureDeviceDiscoverySession *discoverySession = [PAL::getAVCaptureDeviceDiscoverySessionClass()
        discoverySessionWithDeviceTypes:m_avCaptureDeviceTypes.get()
        mediaType:AVMediaTypeVideo
        position:AVCaptureDevicePositionUnspecified
    ];

    return discoverySession.devices;
#else
    return nil;
#endif
}

void AVCaptureDeviceManager::updateCachedAVCaptureDevices()
{
#if HAVE(AVCAPTUREDEVICE)
    ASSERT(!isMainThread());
    auto currentDevices = currentCameras();
    auto removedDevices = adoptNS([[NSMutableArray alloc] init]);
    for (AVCaptureDevice *cachedDevice in m_avCaptureDevices.get()) {
        if (![currentDevices containsObject:cachedDevice])
            [removedDevices addObject:cachedDevice];
    }

    if ([removedDevices count]) {
        for (AVCaptureDevice *device in removedDevices.get())
            [device removeObserver:m_objcObserver.get() forKeyPath:@"suspended"];
        [m_avCaptureDevices removeObjectsInArray:removedDevices.get()];
    }

    for (AVCaptureDevice *device in currentDevices.get()) {

        if (![device hasMediaType:AVMediaTypeVideo] && ![device hasMediaType:AVMediaTypeMuxed])
            continue;

        if ([m_avCaptureDevices containsObject:device])
            continue;

        [device addObserver:m_objcObserver.get() forKeyPath:@"suspended" options:NSKeyValueObservingOptionNew context:(void *)nil];
        [m_avCaptureDevices addObject:device];
    }
#endif
}

static inline CaptureDevice toCaptureDevice(AVCaptureDevice *device, bool isDefault = false)
{
    CaptureDevice captureDevice { device.uniqueID, CaptureDevice::DeviceType::Camera, device.localizedName };
    captureDevice.setEnabled(deviceIsAvailable(device));
    captureDevice.setIsDefault(isDefault);

#if HAVE(AVCAPTUREDEVICE) && HAVE(CONTINUITY_CAMERA)
    if ([PAL::getAVCaptureDeviceClass() respondsToSelector:@selector(systemPreferredCamera)] && [device respondsToSelector:@selector(isContinuityCamera)])
        captureDevice.setIsEphemeral(device.isContinuityCamera && [PAL::getAVCaptureDeviceClass() systemPreferredCamera] != device);
#endif

    return captureDevice;
}

static inline bool isVideoDevice(AVCaptureDevice *device)
{
#if HAVE(AVCAPTUREDEVICE)
    return [device hasMediaType:AVMediaTypeVideo] || [device hasMediaType:AVMediaTypeMuxed];
#else
    UNUSED_PARAM(device);
    return false;
#endif
}

Vector<CaptureDevice> AVCaptureDeviceManager::retrieveCaptureDevices()
{
    ASSERT(!isMainThread());
    if (!isAvailable())
        return { };

    if (!m_avCaptureDevices) {
        m_avCaptureDevices = adoptNS([[NSMutableArray alloc] init]);
        registerForDeviceNotifications();
    }

    updateCachedAVCaptureDevices();

    Vector<CaptureDevice> deviceList;

#if HAVE(AVCAPTUREDEVICE)
    auto currentDevices = currentCameras();
    AVCaptureDevice *defaultVideoDevice = nil;
#if HAVE(CONTINUITY_CAMERA)
    auto haveSystemPreferredCamera = !![PAL::getAVCaptureDeviceClass() respondsToSelector:@selector(systemPreferredCamera)];
    if (haveSystemPreferredCamera)
        defaultVideoDevice = [PAL::getAVCaptureDeviceClass() systemPreferredCamera];
    else
#endif
        defaultVideoDevice = [PAL::getAVCaptureDeviceClass() defaultDeviceWithMediaType: AVMediaTypeVideo];

#if PLATFORM(IOS) || PLATFORM(VISION)
    ([&] {
#if HAVE(CONTINUITY_CAMERA)
        if (haveSystemPreferredCamera && defaultVideoDevice)
            return;
#endif
        if ([defaultVideoDevice position] == AVCaptureDevicePositionFront)
            return;

        defaultVideoDevice = nullptr;
        for (AVCaptureDevice *platformDevice in currentDevices.get()) {
            if (!isVideoDevice(platformDevice))
                continue;

            if ([platformDevice position] == AVCaptureDevicePositionFront) {
                defaultVideoDevice = platformDevice;
                break;
            }
        }
    })();
#endif // PLATFORM(IOS) || PLATFORM(VISION)

    if (defaultVideoDevice)
        deviceList.append(toCaptureDevice(defaultVideoDevice, true));

    for (AVCaptureDevice *platformDevice in currentDevices.get()) {
        if (isVideoDevice(platformDevice) && platformDevice.uniqueID != defaultVideoDevice.uniqueID)
            deviceList.append(toCaptureDevice(platformDevice));
    }
#endif // HAVE(AVCAPTUREDEVICE)

    return deviceList;
}

void AVCaptureDeviceManager::refreshCaptureDevicesInternal(CompletionHandler<void()>&& callback, ShouldSetUserPreferredCamera shouldSetUserPreferredCamera)
{
    m_dispatchQueue->dispatch([this, callback = WTFMove(callback), shouldSetUserPreferredCamera]() mutable {
        if (shouldSetUserPreferredCamera == ShouldSetUserPreferredCamera::Yes)
            setUserPreferredCamera();
        RunLoop::main().dispatch([this, callback = WTFMove(callback), deviceList = crossThreadCopy(retrieveCaptureDevices())]() mutable {
            bool deviceHasChanged = m_devices.size() != deviceList.size();
            if (!deviceHasChanged) {
                for (size_t cptr = 0; cptr < deviceList.size(); ++cptr) {
                    if (m_devices[cptr].persistentId() != deviceList[cptr].persistentId() || m_devices[cptr].enabled() != deviceList[cptr].enabled()) {
                        deviceHasChanged = true;
                        break;
                    }
                }
            }

            if (deviceHasChanged) {
                m_devices = WTFMove(deviceList);
                if (m_isInitialized)
                    deviceChanged();
            }
            callback();
        });
    });
}

bool AVCaptureDeviceManager::isAvailable()
{
    return PAL::isAVFoundationFrameworkAvailable();
}

AVCaptureDeviceManager& AVCaptureDeviceManager::singleton()
{
    static NeverDestroyed<AVCaptureDeviceManager> manager;
    return manager;
}

AVCaptureDeviceManager::AVCaptureDeviceManager()
    : m_objcObserver(adoptNS([[WebCoreAVCaptureDeviceManagerObserver alloc] initWithCallback:this]))
#if HAVE(AVCAPTUREDEVICE)
    , m_avCaptureDeviceTypes(AVVideoCaptureSource::cameraCaptureDeviceTypes())
#endif
    , m_dispatchQueue(WorkQueue::create("com.apple.WebKit.AVCaptureDeviceManager"_s))
{
}

AVCaptureDeviceManager::~AVCaptureDeviceManager()
{
    [[NSNotificationCenter defaultCenter] removeObserver:m_objcObserver.get()];
    [m_objcObserver disconnect];
    for (AVCaptureDevice *device in m_avCaptureDevices.get())
        [device removeObserver:m_objcObserver.get() forKeyPath:@"suspended"];
#if HAVE(AVCAPTUREDEVICE)
    [PAL::getAVCaptureDeviceClass() removeObserver:m_objcObserver.get() forKeyPath:@"systemPreferredCamera"];
    [PAL::getAVCaptureDeviceDiscoverySessionClass() removeObserver:m_objcObserver.get() forKeyPath:@"devices"];
#endif
}

void AVCaptureDeviceManager::setUserPreferredCamera()
{
#if HAVE(AVCAPTUREDEVICE) && PLATFORM(IOS_FAMILY)
    if ([PAL::getAVCaptureDeviceClass() respondsToSelector:@selector(setUserPreferredCamera:)]) {
        auto currentDevices = currentCameras();
        for (AVCaptureDevice *platformDevice in currentDevices.get()) {
            if (isVideoDevice(platformDevice) && [platformDevice position] == AVCaptureDevicePositionFront) {
                [PAL::getAVCaptureDeviceClass() setUserPreferredCamera:platformDevice];
                break;
            }
        }
    }
#endif
}

void AVCaptureDeviceManager::registerForDeviceNotifications()
{
#if HAVE(AVCAPTUREDEVICE)
    [[NSNotificationCenter defaultCenter] addObserver:m_objcObserver.get() selector:@selector(deviceConnectedDidChange:) name:AVCaptureDeviceWasConnectedNotification object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:m_objcObserver.get() selector:@selector(deviceConnectedDidChange:) name:AVCaptureDeviceWasDisconnectedNotification object:nil];
    IGNORE_WARNINGS_BEGIN("objc-method-access")
    [PAL::getAVCaptureDeviceClass() addObserver:m_objcObserver.get() forKeyPath:@"systemPreferredCamera" options:(NSKeyValueObservingOptionOld | NSKeyValueObservingOptionNew) context:nil];
    [PAL::getAVCaptureDeviceDiscoverySessionClass() addObserver:m_objcObserver.get() forKeyPath:@"devices" options:(NSKeyValueObservingOptionOld | NSKeyValueObservingOptionNew) context:nil];
    IGNORE_WARNINGS_END
#endif
}

} // namespace WebCore

@implementation WebCoreAVCaptureDeviceManagerObserver

- (id)initWithCallback:(AVCaptureDeviceManager*)callback
{
    self = [super init];
    if (!self)
        return nil;
    m_callback = callback;
    return self;
}

- (void)disconnect
{
    [NSObject cancelPreviousPerformRequestsWithTarget:self];
    m_callback = nil;
}

- (void)deviceConnectedDidChange:(NSNotification *)unusedNotification
{
    UNUSED_PARAM(unusedNotification);
    if (!m_callback)
        return;

    RunLoop::main().dispatch([self, protectedSelf = retainPtr(self)] {
        if (m_callback)
            m_callback->refreshCaptureDevices();
    });
}

- (void)observeValueForKeyPath:keyPath ofObject:(id)object change:(NSDictionary*)change context:(void*)context
{
    UNUSED_PARAM(object);
    UNUSED_PARAM(context);
    UNUSED_PARAM(change);

    if (!m_callback)
        return;

    if (![keyPath isEqualToString:@"suspended"] && ![keyPath isEqualToString:@"systemPreferredCamera"] && ![keyPath isEqualToString:@"devices"])
        return;

    RunLoop::main().dispatch([self, protectedSelf = retainPtr(self)] {
        if (m_callback)
            m_callback->refreshCaptureDevices();
    });
}

@end

#endif // ENABLE(MEDIA_STREAM) && USE(AVFOUNDATION)
