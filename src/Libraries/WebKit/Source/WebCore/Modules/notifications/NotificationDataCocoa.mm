/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#import "NotificationData.h"

#import "NotificationDirection.h"
#import <wtf/cocoa/VectorCocoa.h>

static NSString * const WebNotificationDefaultActionURLKey = @"WebNotificationDefaultActionURLKey";
static NSString * const WebNotificationTitleKey = @"WebNotificationTitleKey";
static NSString * const WebNotificationBodyKey = @"WebNotificationBodyKey";
static NSString * const WebNotificationIconURLKey = @"WebNotificationIconURLKey";
static NSString * const WebNotificationTagKey = @"WebNotificationTagKey";
static NSString * const WebNotificationLanguageKey = @"WebNotificationLanguageKey";
static NSString * const WebNotificationDirectionKey = @"WebNotificationDirectionKey";
static NSString * const WebNotificationOriginKey = @"WebNotificationOriginKey";
static NSString * const WebNotificationServiceWorkerRegistrationURLKey = @"WebNotificationServiceWorkerRegistrationURLKey";
static NSString * const WebNotificationUUIDStringKey = @"WebNotificationUUIDStringKey";
static NSString * const WebNotificationContextUUIDStringKey = @"WebNotificationContextUUIDStringKey";
static NSString * const WebNotificationSessionIDKey = @"WebNotificationSessionIDKey";
static NSString * const WebNotificationDataKey = @"WebNotificationDataKey";
static NSString * const WebNotificationSilentKey = @"WebNotificationSilentKey";

namespace WebCore {

static std::optional<bool> nsValueToOptionalBool(id value)
{
    if (![value isKindOfClass:[NSNumber class]])
        return std::nullopt;

    return [(NSNumber *)value boolValue];
}

std::optional<NotificationData> NotificationData::fromDictionary(NSDictionary *dictionary)
{
    NSString *defaultActionURL = dictionary[WebNotificationDefaultActionURLKey];
    NSString *title = dictionary[WebNotificationTitleKey];
    NSString *body = dictionary[WebNotificationBodyKey];
    NSString *iconURL = dictionary[WebNotificationIconURLKey];
    NSString *tag = dictionary[WebNotificationTagKey];
    NSString *language = dictionary[WebNotificationLanguageKey];
    NSString *originString = dictionary[WebNotificationOriginKey];
    NSString *serviceWorkerRegistrationURL = dictionary[WebNotificationServiceWorkerRegistrationURLKey];
    NSNumber *sessionID = dictionary[WebNotificationSessionIDKey];
    NSData *notificationData = dictionary[WebNotificationDataKey];

    String uuidString = dictionary[WebNotificationUUIDStringKey];
    auto uuid = WTF::UUID::parseVersion4(uuidString);
    if (!uuid)
        return std::nullopt;

    std::optional<ScriptExecutionContextIdentifier> contextIdentifier;
    String contextUUIDString = dictionary[WebNotificationContextUUIDStringKey];
    if (!contextUUIDString.isEmpty()) {
        auto contextUUID = WTF::UUID::parseVersion4(contextUUIDString);
        if (!contextUUID)
            return std::nullopt;

        contextIdentifier = ScriptExecutionContextIdentifier { *contextUUID, Process::identifier() };
    }

    NotificationDirection direction;
    NSNumber *directionNumber = dictionary[WebNotificationDirectionKey];
    switch ((NotificationDirection)(directionNumber.unsignedLongValue)) {
    case NotificationDirection::Auto:
    case NotificationDirection::Ltr:
    case NotificationDirection::Rtl:
        direction = (NotificationDirection)directionNumber.unsignedLongValue;
        break;
    default:
        return std::nullopt;
    }

    NotificationData data { URL { String { defaultActionURL } }, title, body, iconURL, tag, language, direction, originString, URL { String { serviceWorkerRegistrationURL } }, *uuid, contextIdentifier, PAL::SessionID { sessionID.unsignedLongLongValue }, { }, makeVector(notificationData), nsValueToOptionalBool(dictionary[WebNotificationSilentKey]) };
    return WTFMove(data);
}

NSDictionary *NotificationData::dictionaryRepresentation() const
{
    NSMutableDictionary *result = @{
        WebNotificationDefaultActionURLKey : (NSString *)navigateURL.string(),
        WebNotificationTitleKey : (NSString *)title,
        WebNotificationBodyKey : (NSString *)body,
        WebNotificationIconURLKey : (NSString *)iconURL,
        WebNotificationTagKey : (NSString *)tag,
        WebNotificationLanguageKey : (NSString *)language,
        WebNotificationOriginKey : (NSString *)originString,
        WebNotificationDirectionKey : @((unsigned long)direction),
        WebNotificationServiceWorkerRegistrationURLKey : (NSString *)serviceWorkerRegistrationURL.string(),
        WebNotificationUUIDStringKey : (NSString *)notificationID.toString(),
        WebNotificationSessionIDKey : @(sourceSession.toUInt64()),
        WebNotificationDataKey: toNSData(data).autorelease(),
    }.mutableCopy;

    if (contextIdentifier)
        result[WebNotificationContextUUIDStringKey] = (NSString *)contextIdentifier->toString();

    if (silent != std::nullopt)
        result[WebNotificationSilentKey] = @(*silent);

    return result;
}

} // namespace WebKit
