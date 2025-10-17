/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#pragma once

#include "ExceptionOr.h"
#include "NotificationOptionsPayload.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSDictionary;

namespace WebCore {

struct NotificationData;

struct NotificationPayload {
    URL defaultActionURL;
    String title;
    std::optional<unsigned long long> appBadge;
    std::optional<NotificationOptionsPayload> options;
    bool isMutable { false };

    NotificationPayload isolatedCopy() &&
    {
        return NotificationPayload {
            WTFMove(defaultActionURL).isolatedCopy(),
            WTFMove(title).isolatedCopy(),
            appBadge,
            crossThreadCopy(WTFMove(options)),
            isMutable
        };
    }

#if ENABLE(DECLARATIVE_WEB_PUSH)
    WEBCORE_EXPORT static bool hasDeclarativeMessageHeader(const String& message);
    WEBCORE_EXPORT static ExceptionOr<NotificationPayload> parseJSON(const String&);
    NotificationPayload static fromNotificationData(const NotificationData&);

    WEBCORE_EXPORT NotificationData toNotificationData() const;

#if PLATFORM(COCOA)
    WEBCORE_EXPORT static std::optional<NotificationPayload> fromDictionary(NSDictionary *);
    WEBCORE_EXPORT NSDictionary *dictionaryRepresentation() const;
#endif
#endif // ENABLE(DECLARATIVE_WEB_PUSH)
};

} // namespace WebCore

