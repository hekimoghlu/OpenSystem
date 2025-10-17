/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#import "NotificationPayload.h"

#if ENABLE(DECLARATIVE_WEB_PUSH)

static NSString * const WebNotificationDefaultActionKey = @"WebNotificationDefaultActionKey";
static NSString * const WebNotificationAppBadgeKey = @"WebNotificationAppBadgeKey";
static NSString * const WebNotificationOptionsKey = @"WebNotificationOptionsKey";
static NSString * const WebNotificationMutableKey = @"WebNotificationMutableKey";

namespace WebCore {

std::optional<NotificationPayload> NotificationPayload::fromDictionary(NSDictionary *dictionary)
{
    NSURL *defaultAction = dictionary[WebNotificationDefaultActionKey];
    if (!defaultAction)
        return std::nullopt;

    NSString *title = dictionary[WebNotificationTitleKey];
    if (!title)
        return std::nullopt;

    NSNumber *appBadge = dictionary[WebNotificationAppBadgeKey];
    if (!appBadge)
        return std::nullopt;

    std::optional<uint64_t> rawAppBadge;
    if (![appBadge isKindOfClass:[NSNull class]]) {
        if (![appBadge isKindOfClass:[NSNumber class]])
            return std::nullopt;
        rawAppBadge = [appBadge unsignedLongLongValue];
    }

    NSDictionary *options = dictionary[WebNotificationOptionsKey];
    if (!options)
        return std::nullopt;

    std::optional<NotificationOptionsPayload> rawOptions;
    if (![options isKindOfClass:[NSNull class]]) {
        rawOptions = NotificationOptionsPayload::fromDictionary(options);
        if (!rawOptions)
            return std::nullopt;
    }

    NSNumber *isMutable = dictionary[WebNotificationMutableKey];
    if (!isMutable)
        return std::nullopt;

    return NotificationPayload { defaultAction, title, WTFMove(rawAppBadge), WTFMove(rawOptions), !![isMutable boolValue] };
}

NSDictionary *NotificationPayload::dictionaryRepresentation() const
{
    id nsAppBadge = appBadge ? @(*appBadge) : [NSNull null];
    id nsOptions = options ? options->dictionaryRepresentation() : [NSNull null];

    return @{
        WebNotificationDefaultActionKey : (NSURL *)defaultActionURL,
        WebNotificationTitleKey : (NSString *)title,
        WebNotificationAppBadgeKey : nsAppBadge,
        WebNotificationOptionsKey : nsOptions,
        WebNotificationMutableKey : @(isMutable),
    };
}

} // namespace WebKit

#endif // ENABLE(DECLARATIVE_WEB_PUSH)
