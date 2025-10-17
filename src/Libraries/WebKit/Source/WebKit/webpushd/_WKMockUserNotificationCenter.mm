/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#import "_WKMockUserNotificationCenter.h"

#import <wtf/BlockPtr.h>
#import <wtf/RetainPtr.h>

#if HAVE(FULL_FEATURED_USER_NOTIFICATIONS)

@interface _WKMockUserNotificationCenter ()
- (instancetype)_internalInitWithBundleIdentifier:(NSString *)bundleIdentifier;
@end

static _WKMockUserNotificationCenter *centersByBundleIdentifier(NSString *bundleIdentifier)
{
    static NeverDestroyed<RetainPtr<NSMutableDictionary>> centers = adoptNS([NSMutableDictionary new]);

    if (!centers->get()[bundleIdentifier])
        centers->get()[bundleIdentifier] = adoptNS([[_WKMockUserNotificationCenter alloc] _internalInitWithBundleIdentifier:bundleIdentifier]).autorelease();

    return centers->get()[bundleIdentifier];
}

@implementation _WKMockUserNotificationCenter {
    dispatch_queue_t m_queue;
    BOOL m_hasPermission;
    RetainPtr<NSString> m_bundleIdentifier;
    RetainPtr<NSMutableArray> m_notifications;
    RetainPtr<NSNumber> m_appBadge;
}

- (instancetype)_internalInitWithBundleIdentifier:(NSString *)bundleIdentifier
{
    self = [super init];
    if (!self)
        return nil;

    m_queue = dispatch_queue_create(nullptr, DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
    m_bundleIdentifier = bundleIdentifier;
    m_notifications = adoptNS([[NSMutableArray alloc] init]);

    return self;
}

- (instancetype)initWithBundleIdentifier:(NSString *)bundleIdentifier
{
    self = centersByBundleIdentifier(bundleIdentifier);
    return [self retain];
}

- (void)addNotificationRequest:(UNNotificationRequest *)request withCompletionHandler:(nullable void(^)(NSError *error))completionHandler
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [m_notifications.get() addObject:[UNNotification notificationWithRequest:request date:[NSDate now]]];
#pragma clang diagnostic pop

    // For testing purposes, we know that requests without a targetContentIdentifier are for badging only
    if (!request.content.targetContentIdentifier)
        m_appBadge = request.content.badge;

    dispatch_async(m_queue, ^{
        completionHandler(nil);
    });
}

- (NSNumber *)getAppBadgeForTesting
{
    return m_appBadge.get();
}

- (void)getDeliveredNotificationsWithCompletionHandler:(void(^)(NSArray<UNNotification *> *notifications))completionHandler
{
    RetainPtr<NSArray> notifications = adoptNS([m_notifications copy]);
    dispatch_async(m_queue, ^{
        completionHandler(notifications.get());
    });
}


- (void)removePendingNotificationRequestsWithIdentifiers:(NSArray<NSString *> *) identifiers
{
    RetainPtr toRemove = adoptNS([NSMutableArray new]);
    for (UNNotification *notification in m_notifications.get()) {
        if ([identifiers containsObject:notification.request.identifier])
            [toRemove addObject:notification];
    }

    [m_notifications removeObjectsInArray:toRemove.get()];
}

- (void)removeDeliveredNotificationsWithIdentifiers:(NSArray<NSString *> *) identifiers
{
    // For now, the mock UNUserNotificationCenter doesn't distinguish between pending and delivered notifications.
    [self removePendingNotificationRequestsWithIdentifiers:identifiers];
}

- (void)getNotificationSettingsWithCompletionHandler:(void(^)(UNNotificationSettings *settings))completionHandler
{
    BOOL hasPermission = m_hasPermission;
    dispatch_async(m_queue, ^{
        UNMutableNotificationSettings *settings = [UNMutableNotificationSettings emptySettings];
        settings.authorizationStatus = hasPermission ? UNAuthorizationStatusAuthorized : UNAuthorizationStatusNotDetermined;
        completionHandler(settings);
    });
}

- (void)requestAuthorizationWithOptions:(UNAuthorizationOptions)options completionHandler:(void (^)(BOOL granted, NSError *))completionHandler
{
    m_hasPermission = YES;
    dispatch_async(m_queue, ^{
        completionHandler(YES, nil);
    });
}

- (void)setNotificationCategories:(NSSet<UNNotificationCategory *> *) categories
{
    // No-op. Stubbed out for compatibiltiy with UNUserNotificationCenter.
}

- (UNNotificationSettings *)notificationSettings
{
    RetainPtr settings = [UNMutableNotificationSettings emptySettings];
    [settings setAuthorizationStatus:m_hasPermission ? UNAuthorizationStatusAuthorized : UNAuthorizationStatusNotDetermined];
    return settings.get();
}

@end

#endif // HAVE(FULL_FEATURED_USER_NOTIFICATIONS)
