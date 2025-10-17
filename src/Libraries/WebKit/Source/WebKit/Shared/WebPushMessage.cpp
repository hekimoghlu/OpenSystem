/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#include "WebPushMessage.h"

#include <WebCore/NotificationData.h>
#include <WebCore/SecurityOriginData.h>

namespace WebKit {

#if ENABLE(DECLARATIVE_WEB_PUSH)

WebCore::NotificationData WebPushMessage::notificationPayloadToCoreData() const
{
    RELEASE_ASSERT(notificationPayload);

    static NeverDestroyed<WebCore::ScriptExecutionContextIdentifier> sharedScriptIdentifier = WebCore::ScriptExecutionContextIdentifier::generate();

    String body, iconURL, tag, language;
    auto direction = WebCore::NotificationDirection::Auto;
    std::optional<bool> silent;

    Vector<uint8_t> dataJSON;
    if (notificationPayload->options) {
        body = notificationPayload->options->body;
        language = notificationPayload->options->lang;
        tag = notificationPayload->options->tag;
        iconURL = notificationPayload->options->icon;
        direction = notificationPayload->options->dir;
        silent = notificationPayload->options->silent;

        CString dataCString = notificationPayload->options->dataJSONString.utf8();
        dataJSON = dataCString.span();
    }

    return {
        notificationPayload->defaultActionURL,
        notificationPayload->title,
        WTFMove(body),
        WTFMove(iconURL),
        WTFMove(tag),
        WTFMove(language),
        direction,
        WebCore::SecurityOriginData::fromURL(registrationURL).toString(),
        registrationURL,
        WTF::UUID::createVersion4(),
        sharedScriptIdentifier,
        PAL::SessionID::defaultSessionID(),
        MonotonicTime::now(),
        WTFMove(dataJSON),
        WTFMove(silent)
    };
}

#endif // ENABLE(DECLARATIVE_WEB_PUSH)

} // namespace WebKit
