/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#include "ScriptExecutionContextIdentifier.h"
#include <optional>
#include <pal/SessionID.h>
#include <wtf/MonotonicTime.h>
#include <wtf/URL.h>
#include <wtf/UUID.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSDictionary;

namespace WebCore {

enum class NotificationDirection : uint8_t;

struct NotificationData {
    WEBCORE_EXPORT NotificationData isolatedCopy() const &;
    WEBCORE_EXPORT NotificationData isolatedCopy() &&;

#if PLATFORM(COCOA)
    WEBCORE_EXPORT static std::optional<NotificationData> fromDictionary(NSDictionary *dictionaryRepresentation);
    WEBCORE_EXPORT NSDictionary *dictionaryRepresentation() const;
#endif

    bool isPersistent() const { return !serviceWorkerRegistrationURL.isNull(); }

    URL navigateURL;
    String title;
    String body;
    String iconURL;
    String tag;
    String language;
    WebCore::NotificationDirection direction;
    String originString;
    URL serviceWorkerRegistrationURL;
    WTF::UUID notificationID { WTF::UUID::createVersion4() };
    std::optional<ScriptExecutionContextIdentifier> contextIdentifier;
    PAL::SessionID sourceSession { PAL::SessionID::defaultSessionID() };
    MonotonicTime creationTime;
    Vector<uint8_t> data;
    std::optional<bool> silent;
};

} // namespace WebCore
