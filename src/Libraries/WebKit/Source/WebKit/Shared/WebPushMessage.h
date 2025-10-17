/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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

#include <WebCore/NotificationPayload.h>
#include <optional>
#include <wtf/URL.h>
#include <wtf/Vector.h>


OBJC_CLASS NSDictionary;

namespace WebCore {
struct NotificationData;
}

namespace WebKit {

struct WebPushMessage {
    std::optional<Vector<uint8_t>> pushData;
    String pushPartitionString;
    URL registrationURL;
    std::optional<WebCore::NotificationPayload> notificationPayload;

#if ENABLE(DECLARATIVE_WEB_PUSH)
    WebCore::NotificationData notificationPayloadToCoreData() const;
#endif

#if PLATFORM(COCOA)
    static std::optional<WebPushMessage> fromDictionary(NSDictionary *);
    NSDictionary *toDictionary() const;
#endif
};

} // namespace WebKit
