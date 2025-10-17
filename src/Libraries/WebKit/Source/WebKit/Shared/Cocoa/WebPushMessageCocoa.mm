/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#import "WebPushMessage.h"

#import <wtf/RetainPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {

#define WebKitPushDataKey @"WebKitPushData"
#define WebKitPushRegistrationURLKey @"WebKitPushRegistrationURL"
#define WebKitPushPartitionKey @"WebKitPushPartition"
#define WebKitNotificationPayloadKey @"WebKitNotificationPayload"

std::optional<WebPushMessage> WebPushMessage::fromDictionary(NSDictionary *dictionary)
{
    NSURL *url = [dictionary objectForKey:WebKitPushRegistrationURLKey];
    if (!url || ![url isKindOfClass:[NSURL class]])
        return std::nullopt;

    id pushData = [dictionary objectForKey:WebKitPushDataKey];
    BOOL isNull = [pushData isEqual:[NSNull null]];
    BOOL isData = [pushData isKindOfClass:[NSData class]];

    if (!isNull && !isData)
        return std::nullopt;

    NSString *pushPartition = [dictionary objectForKey:WebKitPushPartitionKey];
    if (!pushPartition || ![pushPartition isKindOfClass:[NSString class]])
        return std::nullopt;

#if ENABLE(DECLARATIVE_WEB_PUSH)
    id payloadDictionary = [dictionary objectForKey:WebKitNotificationPayloadKey];
    isNull = [payloadDictionary isEqual:[NSNull null]];
    BOOL isCorrectType = [payloadDictionary isKindOfClass:[NSDictionary class]];

    if (!isNull && !isCorrectType)
        return std::nullopt;

    std::optional<WebCore::NotificationPayload> payload;
    if (isCorrectType) {
        payload = WebCore::NotificationPayload::fromDictionary(payloadDictionary);
        if (!payload)
            return std::nullopt;
    }

    WebPushMessage message { { }, String { pushPartition }, URL { url }, WTFMove(payload) };
#else
    WebPushMessage message { { }, String { pushPartition }, URL { url }, { } };
#endif

    if (isData)
        message.pushData = makeVector((NSData *)pushData);

    return message;
}

NSDictionary *WebPushMessage::toDictionary() const
{
    RetainPtr<NSData> nsData;
    if (pushData)
        nsData = nsData = toNSData(pushData->span());

    NSDictionary *nsPayload = nil;
#if ENABLE(DECLARATIVE_WEB_PUSH)
    if (notificationPayload)
        nsPayload = notificationPayload->dictionaryRepresentation();
#endif

    return @{
        WebKitPushDataKey : nsData ? nsData.get() : [NSNull null],
        WebKitPushRegistrationURLKey : (NSURL *)registrationURL,
        WebKitPushPartitionKey : (NSString *)pushPartitionString,
        WebKitNotificationPayloadKey : nsPayload ? nsPayload : [NSNull null]
    };
}

} // namespace WebKit
