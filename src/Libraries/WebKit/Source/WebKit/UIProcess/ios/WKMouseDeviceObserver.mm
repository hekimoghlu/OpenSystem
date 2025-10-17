/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#import "WKMouseDeviceObserver.h"

#if HAVE(MOUSE_DEVICE_OBSERVATION)

#import "WebProcessProxy.h"
#import <wtf/BlockPtr.h>
#import <wtf/MainThread.h>
#import <wtf/OSObjectPtr.h>
#import <wtf/RetainPtr.h>

@implementation WKMouseDeviceObserver {
    BOOL _hasMouseDevice;
    size_t _startCount;
    RetainPtr<id<BSInvalidatable>> _token;
    OSObjectPtr<dispatch_queue_t> _deviceObserverTokenQueue;
}

+ (WKMouseDeviceObserver *)sharedInstance
{
    static NeverDestroyed<RetainPtr<WKMouseDeviceObserver>> instance = adoptNS([[WKMouseDeviceObserver alloc] init]);
    return instance.get().get();
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    _deviceObserverTokenQueue = adoptOSObject(dispatch_queue_create("WKMouseDeviceObserver _deviceObserverTokenQueue", DISPATCH_QUEUE_SERIAL));

    return self;
}

#pragma mark - BKSMousePointerDeviceObserver state

- (void)start
{
    [self startWithCompletionHandler:^{ }];
}

- (void)startWithCompletionHandler:(void (^)(void))completionHandler
{
    if (++_startCount > 1)
        return;

    dispatch_async(_deviceObserverTokenQueue.get(), [strongSelf = retainPtr(self), completionHandler = makeBlockPtr(completionHandler)] {
        ASSERT(!strongSelf->_token);
        strongSelf->_token = [[BKSMousePointerService sharedInstance] addPointerDeviceObserver:strongSelf.get()];

        completionHandler();
    });
}

- (void)stop
{
    [self stopWithCompletionHandler:^{ }];
}

- (void)stopWithCompletionHandler:(void (^)(void))completionHandler
{
    ASSERT(_startCount);
    if (!_startCount || --_startCount)
        return;

    dispatch_async(_deviceObserverTokenQueue.get(), [strongSelf = retainPtr(self), completionHandler = makeBlockPtr(completionHandler)] {
        ASSERT(strongSelf->_token);
        [strongSelf->_token invalidate];
        strongSelf->_token = nil;

        completionHandler();
    });
}

#pragma mark - BKSMousePointerDeviceObserver handlers

- (void)mousePointerDevicesDidChange:(NSSet<BKSMousePointerDevice *> *)mousePointerDevices
{
    BOOL hasMouseDevice = mousePointerDevices.count > 0;
    if (hasMouseDevice == _hasMouseDevice)
        return;

    _hasMouseDevice = hasMouseDevice;

    ensureOnMainRunLoop([hasMouseDevice] {
        WebKit::WebProcessProxy::notifyHasMouseDeviceChanged(hasMouseDevice);
    });
}

#pragma mark - Testing

- (void)_setHasMouseDeviceForTesting:(BOOL)hasMouseDevice
{
    _hasMouseDevice = hasMouseDevice;

    ensureOnMainRunLoop([hasMouseDevice] {
        WebKit::WebProcessProxy::notifyHasMouseDeviceChanged(hasMouseDevice);
    });
}

@end

#endif // HAVE(MOUSE_DEVICE_OBSERVATION)
