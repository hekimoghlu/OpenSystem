/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#import "WKStylusDeviceObserver.h"

#if HAVE(STYLUS_DEVICE_OBSERVATION)

#import "WebProcessProxy.h"
#import <UIKit/UIScribbleInteraction.h>
#import <wtf/RetainPtr.h>
#import <wtf/Seconds.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

static Seconds changeTimeInterval { 10_min };

@implementation WKStylusDeviceObserver {
    BOOL _hasStylusDevice;
    size_t _startCount;
    RetainPtr<NSTimer> _changeTimer;
}

+ (WKStylusDeviceObserver *)sharedInstance
{
    static NeverDestroyed<RetainPtr<WKStylusDeviceObserver>> instance = adoptNS([[WKStylusDeviceObserver alloc] init]);
    return instance.get().get();
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    _hasStylusDevice = UIScribbleInteraction.isPencilInputExpected;

    if (NSNumber *changeTimeIntervalOverride = [NSUserDefaults.standardUserDefaults objectForKey:@"WKStylusDeviceObserverChangeTimeInterval"]) {
        changeTimeInterval = Seconds([changeTimeIntervalOverride doubleValue]);
        WTFLogAlways("Warning: WKStylusDeviceObserver changeTimeInterval was overriden via user defaults and is now %g seconds", changeTimeInterval.seconds());
    }

    return self;
}

#pragma mark - State

- (void)setHasStylusDevice:(BOOL)hasStylusDevice
{
    if (hasStylusDevice != _hasStylusDevice) {
        _hasStylusDevice = hasStylusDevice;

        WebKit::WebProcessProxy::notifyHasStylusDeviceChanged(_hasStylusDevice);
    }

    [_changeTimer invalidate];
    _changeTimer = nil;
}

- (void)start
{
    if (++_startCount > 1)
        return;

    if (linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::ObservesClassProperty))
        [[UIScribbleInteraction class] addObserver:self forKeyPath:@"isPencilInputExpected" options:NSKeyValueObservingOptionInitial context:nil];
}

- (void)stop
{
    ASSERT(_startCount);
    if (!_startCount || --_startCount)
        return;

    if (linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::ObservesClassProperty))
        [[UIScribbleInteraction class] removeObserver:self forKeyPath:@"isPencilInputExpected"];
}

#pragma mark - isPencilInputExpected KVO

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context
{
    ASSERT([keyPath isEqualToString:@"isPencilInputExpected"]);
    ASSERT(object == [UIScribbleInteraction class]);

    if (UIScribbleInteraction.isPencilInputExpected)
        self.hasStylusDevice = YES;
    else
        [self startChangeTimer:changeTimeInterval.seconds()];
}

#pragma mark - isPencilInputExpected Timer

- (void)startChangeTimer:(NSTimeInterval)timeInterval
{
    [_changeTimer invalidate];
    _changeTimer = [NSTimer scheduledTimerWithTimeInterval:timeInterval target:self selector:@selector(changeTimerFired:) userInfo:nil repeats:NO];
}

- (void)changeTimerFired:(NSTimer *)timer
{
    self.hasStylusDevice = NO;
}

@end

#endif // HAVE(STYLUS_DEVICE_OBSERVATION)
