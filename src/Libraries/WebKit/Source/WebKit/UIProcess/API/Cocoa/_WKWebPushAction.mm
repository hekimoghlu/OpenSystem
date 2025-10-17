/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
#import "_WKWebPushActionInternal.h"

#import "Logging.h"
#import "UserNotificationsSPI.h"
#import "WebPushDaemonConstants.h"

#if PLATFORM(IOS)
#import "UIKitSPI.h"
#endif

NSString * const _WKWebPushActionTypePushEvent = @"_WKWebPushActionTypePushEvent";
NSString * const _WKWebPushActionTypeNotificationClick = @"_WKWebPushActionTypeNotificationClick";
NSString * const _WKWebPushActionTypeNotificationClose = @"_WKWebPushActionTypeNotificationClose";

@interface _WKWebPushAction ()
@property (nonatomic, readwrite, strong) NSNumber *version;
@property (nonatomic, readwrite, strong) NSUUID *webClipIdentifier;
@property (nonatomic, readwrite, strong) NSString *type;
@property (nonatomic, readwrite, assign) std::optional<WebCore::NotificationData> coreNotificationData;
@end

@implementation _WKWebPushAction

static RetainPtr<NSUUID> uuidFromPushPartition(NSString *pushPartition)
{
    if (pushPartition.length != 32)
        return nil;

    RetainPtr uuidString = [NSString stringWithFormat:@"%@-%@-%@-%@-%@",
        [pushPartition substringWithRange:NSMakeRange(0, 8)],
        [pushPartition substringWithRange:NSMakeRange(8, 4)],
        [pushPartition substringWithRange:NSMakeRange(12, 4)],
        [pushPartition substringWithRange:NSMakeRange(16, 4)],
        [pushPartition substringWithRange:NSMakeRange(20, 12)]];
    return adoptNS([[NSUUID alloc] initWithUUIDString:uuidString.get()]);
}

- (void)dealloc
{
    [_version release];
    [_webClipIdentifier release];
    [_type release];
    [super dealloc];
}

+ (_WKWebPushAction *)webPushActionWithDictionary:(NSDictionary *)dictionary
{
    NSNumber *version = dictionary[WebKit::WebPushD::pushActionVersionKey()];
    if (!version || ![version isKindOfClass:[NSNumber class]])
        return nil;

    NSString *pushPartition = dictionary[WebKit::WebPushD::pushActionPartitionKey()];
    if (!pushPartition || ![pushPartition isKindOfClass:[NSString class]])
        return nil;

    RetainPtr uuid = uuidFromPushPartition(pushPartition);
    if (!uuid)
        return nil;

    NSString *type = dictionary[WebKit::WebPushD::pushActionTypeKey()];
    if (!type || ![type isKindOfClass:[NSString class]])
        return nil;

    _WKWebPushAction *result = [[_WKWebPushAction alloc] init];
    result.version = version;
    result.webClipIdentifier = uuid.get();
    result.type = type;

    return [result autorelease];
}

+ (_WKWebPushAction *)_webPushActionWithNotificationResponse:(UNNotificationResponse *)response
{
#if PLATFORM(IOS)
    if (![response.notification.sourceIdentifier hasPrefix:@"com.apple.WebKit.PushBundle."])
        return nil;

    NSString *pushPartition = [response.notification.sourceIdentifier substringFromIndex:28];
    RetainPtr webClipIdentifier = uuidFromPushPartition(pushPartition);
    if (!webClipIdentifier)
        return nil;

    auto notificationData = WebCore::NotificationData::fromDictionary(response.notification.request.content.userInfo);
    if (!notificationData)
        return nil;

    _WKWebPushAction *result = [[[_WKWebPushAction alloc] init] autorelease];
    result.webClipIdentifier = webClipIdentifier.get();

    if ([response.actionIdentifier isEqualToString:UNNotificationDefaultActionIdentifier])
        result.type = _WKWebPushActionTypeNotificationClick;
    else if ([response.actionIdentifier isEqualToString:UNNotificationDismissActionIdentifier])
        result.type = _WKWebPushActionTypeNotificationClose;
    else {
        RELEASE_LOG_ERROR(Push, "Unknown notification response action identifier encountered: %@", response.actionIdentifier);
        return nil;
    }

    result.coreNotificationData = WTFMove(notificationData);

    return result;
#else
    return nil;
#endif // PLATFORM(IOS)
}

- (NSString *)_nameForBackgroundTaskAndLogging
{
    if ([_type isEqualToString:_WKWebPushActionTypePushEvent])
        return @"Web Push Event";
    if ([_type isEqualToString:_WKWebPushActionTypeNotificationClick])
        return @"Web Notification Click";
    if ([_type isEqualToString:_WKWebPushActionTypeNotificationClose])
        return @"Web Notification Close";

    return @"Unknown Web Push event";
}

- (UIBackgroundTaskIdentifier)beginBackgroundTaskForHandling
{
#if PLATFORM(IOS)
    NSString *taskName;
    if (_webClipIdentifier)
        taskName = [NSString stringWithFormat:@"%@ for %@", self._nameForBackgroundTaskAndLogging, _webClipIdentifier];
    else
        taskName = [NSString stringWithFormat:@"%@", self._nameForBackgroundTaskAndLogging];

    return [UIApplication.sharedApplication beginBackgroundTaskWithName:taskName expirationHandler:^{
        RELEASE_LOG_ERROR(Push, "Took too long to handle Web Push action: '%@'", taskName);
    }];
#else
    return 0;
#endif // PLATFORM(IOS)
}

@end
