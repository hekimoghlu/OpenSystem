/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#import "NotificationOptionsPayload.h"

#if ENABLE(DECLARATIVE_WEB_PUSH)

static NSString * const WebDirKey = @"WebDirKey";
static NSString * const WebLangKey = @"WebLangKey";
static NSString * const WebBodyKey = @"WebBodyKey";
static NSString * const WebTagKey = @"WebTagKey";
static NSString * const WebIconKey = @"WebIconKey";
static NSString * const WebDataJSONKey = @"WebDataJSONKey";
static NSString * const WebSilentKey = @"WebSilentKey";

namespace WebCore {

std::optional<NotificationOptionsPayload> NotificationOptionsPayload::fromDictionary(NSDictionary *dictionary)
{
    if (![dictionary isKindOfClass:[NSDictionary class]])
        return std::nullopt;

    NSNumber *dir = dictionary[WebDirKey];
    if (![dir isKindOfClass:[NSNumber class]])
        return std::nullopt;

    auto dirValue = [dir unsignedCharValue];
    if (!isValidNotificationDirection(dirValue))
        return std::nullopt;
    auto rawDir = (NotificationDirection)dirValue;

    NSString *lang = dictionary[WebLangKey];
    NSString *body = dictionary[WebBodyKey];
    NSString *tag = dictionary[WebTagKey];
    NSString *icon = dictionary[WebIconKey];
    NSString *dataJSON = dictionary[WebDataJSONKey];

    NSNumber *silent = dictionary[WebSilentKey];
    if (!silent)
        return std::nullopt;

    std::optional<bool> rawSilent;
    if (![silent isKindOfClass:[NSNull class]]) {
        if (![silent isKindOfClass:[NSNumber class]])
            return std::nullopt;

        rawSilent = [silent boolValue];
    }

    return NotificationOptionsPayload { (NotificationDirection)rawDir, lang, body, tag, icon, dataJSON, rawSilent };
}

NSDictionary *NotificationOptionsPayload::dictionaryRepresentation() const
{
    return @{
        WebDirKey : @((uint8_t)dir),
        WebLangKey : (NSString *)lang,
        WebBodyKey : (NSString *)body,
        WebTagKey : (NSString *)tag,
        WebIconKey : (NSString *)icon,
        WebDataJSONKey : (NSString *)dataJSONString,
        WebSilentKey : silent.has_value() ? @(*silent) : [NSNull null],
    };
}

} // namespace WebKit

#endif // ENABLE(DECLARATIVE_WEB_PUSH)
