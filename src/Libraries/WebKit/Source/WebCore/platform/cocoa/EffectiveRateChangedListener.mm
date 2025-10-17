/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#import "EffectiveRateChangedListener.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CMTime.h>
#import <pal/spi/cf/CFNotificationCenterSPI.h>
#import <wtf/Function.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/spi/cocoa/NSObjCRuntimeSPI.h>

#import <pal/cf/CoreMediaSoftLink.h>

@interface WebEffectiveRateChangedListenerObjCAdapter : NSObject
@property (atomic, readonly, direct) RefPtr<WebCore::EffectiveRateChangedListener> protectedListener;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithEffectiveRateChangedListener:(const WebCore::EffectiveRateChangedListener&)listener;
@end

NS_DIRECT_MEMBERS
@implementation WebEffectiveRateChangedListenerObjCAdapter {
    ThreadSafeWeakPtr<WebCore::EffectiveRateChangedListener> _listener;
}

- (instancetype)initWithEffectiveRateChangedListener:(const WebCore::EffectiveRateChangedListener&)listener
{
    if ((self = [super init]))
        _listener = listener;
    return self;
}

- (RefPtr<WebCore::EffectiveRateChangedListener>)protectedListener
{
    return _listener.get();
}
@end

namespace WebCore {

static void timebaseEffectiveRateChangedCallback(CFNotificationCenterRef, void* observer, CFNotificationName, const void*, CFDictionaryRef)
{
    RetainPtr adapter { dynamic_objc_cast<WebEffectiveRateChangedListenerObjCAdapter>(reinterpret_cast<id>(observer)) };
    if (RefPtr protectedListener = [adapter protectedListener])
        protectedListener->effectiveRateChanged();
}

EffectiveRateChangedListener::EffectiveRateChangedListener(Function<void()>&& callback, CMTimebaseRef timebase)
    : m_callback(WTFMove(callback))
    , m_objcAdapter(adoptNS([[WebEffectiveRateChangedListenerObjCAdapter alloc] initWithEffectiveRateChangedListener:*this]))
    , m_timebase(timebase)
{
    assertIsMainThread();
    ASSERT(timebase);
    // Observer removed MediaPlayerPrivateMediaSourceAVFObjC destructor.
    CFNotificationCenterAddObserver(CFNotificationCenterGetLocalCenter(), m_objcAdapter.get(), timebaseEffectiveRateChangedCallback, kCMTimebaseNotification_EffectiveRateChanged, timebase, static_cast<CFNotificationSuspensionBehavior>(_CFNotificationObserverIsObjC));
}

EffectiveRateChangedListener::~EffectiveRateChangedListener()
{
    stop();
}

void EffectiveRateChangedListener::effectiveRateChanged()
{
    m_callback();
}

void EffectiveRateChangedListener::stop()
{
    assertIsMainThread();
    if (!m_timebase)
        return;
    CFNotificationCenterRemoveObserver(CFNotificationCenterGetLocalCenter(), m_objcAdapter.get(), kCMTimebaseNotification_EffectiveRateChanged, m_timebase.get());
    m_timebase = nullptr;
}

} // namespace WebCore
