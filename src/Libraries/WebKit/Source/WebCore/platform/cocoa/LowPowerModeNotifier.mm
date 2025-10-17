/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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

#if HAVE(APPLE_LOW_POWER_MODE_SUPPORT)
#import "LowPowerModeNotifier.h"

#import "Logging.h"
#import <Foundation/NSProcessInfo.h>
#import <wtf/MainThread.h>

@interface WebLowPowerModeObserver : NSObject
@property (nonatomic) CheckedPtr<WebCore::LowPowerModeNotifier> notifier;
@property (nonatomic, readonly) BOOL isLowPowerModeEnabled;
@end

@implementation WebLowPowerModeObserver {
}

- (WebLowPowerModeObserver *)initWithNotifier:(WebCore::LowPowerModeNotifier&)notifier
{
    self = [super init];
    if (!self)
        return nil;

    _notifier = &notifier;
    _isLowPowerModeEnabled = [NSProcessInfo processInfo].lowPowerModeEnabled;
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(_didReceiveLowPowerModeChange) name:NSProcessInfoPowerStateDidChangeNotification object:nil];
    return self;
}

- (void)dealloc
{
    if (_notifier)
        [self detach];
    [super dealloc];
}

- (void)detach
{
    ASSERT(isMainThread());
    [[NSNotificationCenter defaultCenter] removeObserver:self name:NSProcessInfoPowerStateDidChangeNotification object:nil];
    _notifier = nullptr;
}

- (void)_didReceiveLowPowerModeChange
{
    // We need to make sure we notify the client on the main thread.
    ensureOnMainRunLoop([self, protectedSelf = RetainPtr<WebLowPowerModeObserver>(self), lowPowerModeEnabled = [NSProcessInfo processInfo].lowPowerModeEnabled] {
        if (!_notifier)
            return;
        _isLowPowerModeEnabled = lowPowerModeEnabled;
        notifyLowPowerModeChanged(*_notifier, _isLowPowerModeEnabled);
    });
}

@end

namespace WebCore {

LowPowerModeNotifier::LowPowerModeNotifier(LowPowerModeChangeCallback&& callback)
    : m_observer(adoptNS([[WebLowPowerModeObserver alloc] initWithNotifier:*this]))
    , m_callback(WTFMove(callback))
{
    ASSERT(isMainThread());
}

LowPowerModeNotifier::~LowPowerModeNotifier()
{
    ASSERT(isMainThread());
    [m_observer detach];
}

bool LowPowerModeNotifier::isLowPowerModeEnabled() const
{
    return m_observer.get().isLowPowerModeEnabled;
}

void LowPowerModeNotifier::notifyLowPowerModeChanged(bool enabled)
{
    m_callback(enabled);
}

void notifyLowPowerModeChanged(LowPowerModeNotifier& notifier, bool enabled)
{
    RELEASE_LOG(PerformanceLogging, "Low power mode state has changed to %d", enabled);
    notifier.notifyLowPowerModeChanged(enabled);
}

} // namespace WebCore

#endif // HAVE(APPLE_LOW_POWER_MODE_SUPPORT)
