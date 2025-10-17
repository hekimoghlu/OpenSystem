/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "JSWebExtensionAPIAlarms.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;

namespace WebKit {

class WebExtensionAPIAlarms : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIAlarms, alarms, alarms);

public:
#if PLATFORM(COCOA)
    void createAlarm(NSString *name, NSDictionary *alarmInfo, NSString **outExceptionString);

    void get(NSString *name, Ref<WebExtensionCallbackHandler>&&);
    void getAll(Ref<WebExtensionCallbackHandler>&&);

    void clear(NSString *name, Ref<WebExtensionCallbackHandler>&&);
    void clearAll(Ref<WebExtensionCallbackHandler>&&);

    WebExtensionAPIEvent& onAlarm();

private:
    RefPtr<WebExtensionAPIEvent> m_onAlarm;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
