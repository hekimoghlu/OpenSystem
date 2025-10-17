/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include "config.h"
#include "NotificationPayload.h"

#include "NotificationData.h"
#include "NotificationJSONParser.h"

#if ENABLE(DECLARATIVE_WEB_PUSH)

#include "Logging.h"

namespace WebCore {

bool NotificationPayload::hasDeclarativeMessageHeader(const String& json)
{
    return NotificationJSONParser::hasDeclarativeMessageHeader(json);
}

ExceptionOr<NotificationPayload> NotificationPayload::parseJSON(const String& json)
{
    auto value = JSON::Value::parseJSON(json);
    if (!value)
        return Exception { ExceptionCode::SyntaxError, "Push message with Notification disposition: does not contain valid JSON"_s };

    auto object = value->asObject();
    if (!object)
        return Exception { ExceptionCode::SyntaxError, "Push message with Notification disposition: top level JSON value is not an object"_s };

    return NotificationJSONParser::parseNotificationPayload(*object);
}

NotificationPayload NotificationPayload::fromNotificationData(const NotificationData& data)
{
    NotificationOptionsPayload options { data.direction, data.language, data.body, data.tag, data.iconURL, { }, data.silent };

    return { data.navigateURL, data.title, std::nullopt, WTFMove(options), false };
}

NotificationData NotificationPayload::toNotificationData() const
{
    NotificationData data;
    data.navigateURL = defaultActionURL;
    data.title = title;

    if (options) {
        data.direction = options->dir;
        data.language = options->lang;
        data.body = options->body;
        data.tag = options->tag;
        data.iconURL = options->icon;
        data.silent = options->silent;
    }

    return data;
}

} // namespace WebCore

#endif // ENABLE(DECLARATIVE_WEB_PUSH)
