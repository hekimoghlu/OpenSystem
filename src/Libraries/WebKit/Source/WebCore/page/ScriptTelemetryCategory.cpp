/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#include "ScriptTelemetryCategory.h"

#if !(USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/ScriptTelemetryCategoryAdditions.cpp>))
#include <wtf/URL.h>
#include <wtf/text/MakeString.h>
#endif

namespace WebCore {

ASCIILiteral description(ScriptTelemetryCategory category)
{
    switch (category) {
    case ScriptTelemetryCategory::Unspecified:
        ASSERT_NOT_REACHED();
        return "Unspecified"_s;
    case ScriptTelemetryCategory::Audio:
        return "Audio"_s;
    case ScriptTelemetryCategory::Canvas:
        return "Canvas"_s;
    case ScriptTelemetryCategory::Cookies:
        return "Cookies"_s;
    case ScriptTelemetryCategory::Gamepads:
        return "Gamepads"_s;
    case ScriptTelemetryCategory::HardwareConcurrency:
        return "HardwareConcurrency"_s;
    case ScriptTelemetryCategory::LocalStorage:
        return "LocalStorage"_s;
    case ScriptTelemetryCategory::MediaDevices:
        return "MediaDevices"_s;
    case ScriptTelemetryCategory::Notifications:
        return "Notifications"_s;
    case ScriptTelemetryCategory::Payments:
        return "Payments"_s;
    case ScriptTelemetryCategory::Permissions:
        return "Permissions"_s;
    case ScriptTelemetryCategory::QueryParameters:
        return "QueryParameters"_s;
    case ScriptTelemetryCategory::Referrer:
        return "Referrer"_s;
    case ScriptTelemetryCategory::ScreenOrViewport:
        return "ScreenOrViewport"_s;
    case ScriptTelemetryCategory::Speech:
        return "Speech"_s;
    case ScriptTelemetryCategory::FormControls:
        return "FormControls"_s;
    }
    ASSERT_NOT_REACHED();
    return { };
}

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/ScriptTelemetryCategoryAdditions.cpp>)
#import <WebKitAdditions/ScriptTelemetryCategoryAdditions.cpp>
#else
String makeLogMessage(const URL& url, ScriptTelemetryCategory category)
{
    return makeString(url.string(), " tried to access "_s, description(category));
}
#endif

} // namespace WebCore
