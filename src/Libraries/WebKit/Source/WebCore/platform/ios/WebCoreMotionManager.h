/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#if PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)

#import <CoreLocation/CoreLocation.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakHashSet.h>

constexpr float kMotionUpdateInterval = 1.0f / 60.0f;
@class CMMotionManager;

namespace WebCore {
class DeviceMotionClientIOS;
class MotionManagerClient;
}

WEBCORE_EXPORT @interface WebCoreMotionManager : NSObject {
    RetainPtr<CMMotionManager> m_motionManager;
    RetainPtr<CLLocationManager> m_locationManager;
    WeakHashSet<WebCore::MotionManagerClient> m_deviceMotionClients;
    WeakHashSet<WebCore::MotionManagerClient> m_deviceOrientationClients;
    RetainPtr<NSTimer> m_updateTimer;
    BOOL m_gyroAvailable;
    BOOL m_headingAvailable;
    BOOL m_initialized;
}

+ (WebCoreMotionManager *)sharedManager;
- (void)addMotionClient:(WebCore::MotionManagerClient *)client;
- (void)removeMotionClient:(WebCore::MotionManagerClient *)client;
- (void)addOrientationClient:(WebCore::MotionManagerClient *)client;
- (void)removeOrientationClient:(WebCore::MotionManagerClient *)client;
- (BOOL)gyroAvailable;
- (BOOL)headingAvailable;
@end

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DEVICE_ORIENTATION)
