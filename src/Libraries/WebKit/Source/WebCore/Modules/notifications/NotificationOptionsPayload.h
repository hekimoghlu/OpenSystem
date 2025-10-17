/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#include "NotificationDirection.h"

OBJC_CLASS NSDictionary;

namespace WebCore {

struct NotificationOptionsPayload {
    NotificationDirection dir;
    String lang;
    String body;
    String tag;
    String icon;
    String dataJSONString;
    std::optional<bool> silent;

    NotificationOptionsPayload isolatedCopy() &&
    {
        return NotificationOptionsPayload {
            dir,
            WTFMove(lang).isolatedCopy(),
            WTFMove(body).isolatedCopy(),
            WTFMove(tag).isolatedCopy(),
            WTFMove(icon).isolatedCopy(),
            WTFMove(dataJSONString).isolatedCopy(),
            silent
        };
    }

#if ENABLE(DECLARATIVE_WEB_PUSH)
#if PLATFORM(COCOA)
    static std::optional<NotificationOptionsPayload> fromDictionary(NSDictionary *);
    NSDictionary *dictionaryRepresentation() const;
#endif
#endif // ENABLE(DECLARATIVE_WEB_PUSH)
};

} // namespace WebCore
