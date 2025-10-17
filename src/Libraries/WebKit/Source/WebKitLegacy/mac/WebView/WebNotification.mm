/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#import "WebNotification.h"

#import "WebNotificationInternal.h"

#if ENABLE(NOTIFICATIONS)
#import "WebSecurityOriginInternal.h"
#import <WebCore/Notification.h>
#import <WebCore/NotificationData.h>
#import <WebCore/ScriptExecutionContext.h>
#import <wtf/RefPtr.h>

using namespace WebCore;
#endif

@interface WebNotificationPrivate : NSObject
{
@public
#if ENABLE(NOTIFICATIONS)
    std::optional<NotificationData> _internal;
#endif
}
@end

@implementation WebNotificationPrivate
@end

#if ENABLE(NOTIFICATIONS)
@implementation WebNotification (WebNotificationInternal)

- (id)initWithCoreNotification:(NotificationData&&)coreNotification
{
    if (!(self = [super init]))
        return nil;
    _private = [[WebNotificationPrivate alloc] init];
    _private->_internal = WTFMove(coreNotification);
    return self;
}
@end
#endif

@implementation WebNotification
- (id)init
{
    return nil;
}

- (void)dealloc
{
    [_private release];
    [super dealloc];
}

- (NSString *)title
{
#if ENABLE(NOTIFICATIONS)
    return _private->_internal->title;
#else
    return nil;
#endif
}

- (NSString *)body
{
#if ENABLE(NOTIFICATIONS)
    return _private->_internal->body;
#else
    return nil;
#endif
}

- (NSString *)tag
{
#if ENABLE(NOTIFICATIONS)
    return _private->_internal->tag;
#else
    return nil;
#endif
}

- (NSString *)iconURL
{
#if ENABLE(NOTIFICATIONS)
    return _private->_internal->iconURL;
#else
    return nil;
#endif
}

- (NSString *)lang
{
#if ENABLE(NOTIFICATIONS)
    return _private->_internal->language;
#else
    return nil;
#endif
}

- (NSString *)dir
{
#if ENABLE(NOTIFICATIONS)
    switch (_private->_internal->direction) {
        case Notification::Direction::Auto:
            return @"auto";
        case Notification::Direction::Ltr:
            return @"ltr";
        case Notification::Direction::Rtl:
            return @"rtl";
    }
#else
    return nil;
#endif
}

- (WebSecurityOrigin *)origin
{
#if ENABLE(NOTIFICATIONS)
    return adoptNS([[WebSecurityOrigin alloc] _initWithString:_private->_internal->originString]).autorelease();
#else
    return nil;
#endif
}

- (NSString *)notificationID
{
#if ENABLE(NOTIFICATIONS)
    return _private->_internal->notificationID.toString();
#else
    return 0;
#endif
}

- (void)dispatchShowEvent
{
#if ENABLE(NOTIFICATIONS)
    Notification::ensureOnNotificationThread(*_private->_internal, [](auto* notification) {
        if (notification)
            notification->dispatchShowEvent();
    });
#endif
}

- (void)dispatchCloseEvent
{
#if ENABLE(NOTIFICATIONS)
    Notification::ensureOnNotificationThread(*_private->_internal, [](auto* notification) {
        if (notification)
            notification->dispatchCloseEvent();
    });
#endif
}

- (void)dispatchClickEvent
{
#if ENABLE(NOTIFICATIONS)
    Notification::ensureOnNotificationThread(*_private->_internal, [](auto* notification) {
        if (notification)
            notification->dispatchClickEvent();
    });
#endif
}

- (void)dispatchErrorEvent
{
#if ENABLE(NOTIFICATIONS)
    Notification::ensureOnNotificationThread(*_private->_internal, [](auto* notification) {
        if (notification)
            notification->dispatchErrorEvent();
    });
#endif
}

- (void)finalize
{
#if ENABLE(NOTIFICATIONS)
    Notification::ensureOnNotificationThread(*_private->_internal, [](auto* notification) {
        if (notification)
            notification->finalize();
    });
#endif
}

@end

