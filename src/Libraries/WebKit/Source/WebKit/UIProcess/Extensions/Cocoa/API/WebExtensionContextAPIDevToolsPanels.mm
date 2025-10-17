/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)

#import "APIInspectorExtension.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionUtilities.h"

namespace WebKit {

void WebExtensionContext::devToolsPanelsCreate(WebPageProxyIdentifier webPageProxyIdentifier, const String& title, const String& iconPath, const String& pagePath, CompletionHandler<void(Expected<Inspector::ExtensionTabID, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"devtools.panels.create()";

    RefPtr extension = inspectorExtension(webPageProxyIdentifier);
    if (!extension) {
        RELEASE_LOG_ERROR(Extensions, "Inspector extension not found for page %llu", webPageProxyIdentifier.toUInt64());
        completionHandler(toWebExtensionError(apiName, nullString(), @"Web Inspector not found"));
        return;
    }

    extension->createTab(title, { baseURL(), iconPath }, { baseURL(), pagePath }, [completionHandler = WTFMove(completionHandler)](Expected<Inspector::ExtensionTabID, Inspector::ExtensionError>&& result) mutable {
        if (!result) {
            RELEASE_LOG_ERROR(Extensions, "Inspector could not create panel (%{public}@)", (NSString *)extensionErrorToString(result.error()));
            completionHandler(toWebExtensionError(apiName, nullString(), @"Web Inspector could not create the panel"));
            return;
        }

        completionHandler(result.value());
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
