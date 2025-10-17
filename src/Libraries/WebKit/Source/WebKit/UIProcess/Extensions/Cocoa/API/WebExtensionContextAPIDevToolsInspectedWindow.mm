/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#import "APISerializedScriptValue.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionUtilities.h"
#import <WebCore/ExceptionDetails.h>

namespace WebKit {

void WebExtensionContext::devToolsInspectedWindowEval(WebPageProxyIdentifier webPageProxyIdentifier, const String& scriptSource, const std::optional<URL>& frameURL, CompletionHandler<void(Expected<Expected<std::span<const uint8_t>, WebCore::ExceptionDetails>, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"devtools.inspectedWindow.eval()";

    RefPtr extension = inspectorExtension(webPageProxyIdentifier);
    if (!extension) {
        RELEASE_LOG_ERROR(Extensions, "Inspector extension not found for page %llu", webPageProxyIdentifier.toUInt64());
        completionHandler(toWebExtensionError(apiName, nullString(), @"Web Inspector not found"));
        return;
    }

    // FIXME: <https://webkit.org/b/269349> Implement `contextSecurityOrigin` and `useContentScriptContext` options for `devtools.inspectedWindow.eval` command

    RefPtr tab = getTab(webPageProxyIdentifier, std::nullopt, IncludeExtensionViews::Yes);
    if (!tab) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"tab not found"));
        return;
    }

    requestPermissionToAccessURLs({ tab->url() }, tab, [extension, tab, scriptSource, frameURL, completionHandler = WTFMove(completionHandler)](auto&& requestedURLs, auto&& allowedURLs, auto expirationDate) mutable {
        if (!tab->extensionHasPermission()) {
            completionHandler(toWebExtensionError(apiName, nullString(), @"this extension does not have access to this tab"));
            return;
        }

        extension->evaluateScript(scriptSource, frameURL, std::nullopt, std::nullopt, [completionHandler = WTFMove(completionHandler)](Inspector::ExtensionEvaluationResult&& result) mutable {
            if (!result) {
                RELEASE_LOG_ERROR(Extensions, "Inspector could not evaluate script (%{public}@)", (NSString *)extensionErrorToString(result.error()));
                completionHandler(toWebExtensionError(apiName, nullString(), @"Web Inspector could not evaluate script"));
                return;
            }

            if (!result.value()) {
                Expected<std::span<const uint8_t>, WebCore::ExceptionDetails> returnedValue = makeUnexpected(result.value().error());
                completionHandler({ WTFMove(returnedValue) });
                return;
            }

            completionHandler({ result.value()->get().dataReference() });
        });
    });
}

void WebExtensionContext::devToolsInspectedWindowReload(WebPageProxyIdentifier webPageProxyIdentifier, const std::optional<bool>& ignoreCache)
{
    RefPtr extension = inspectorExtension(webPageProxyIdentifier);
    if (!extension) {
        RELEASE_LOG_ERROR(Extensions, "Inspector extension not found for page %llu", webPageProxyIdentifier.toUInt64());
        return;
    }

    // FIXME: <https://webkit.org/b/222328> Implement `userAgent` and `injectedScript` options for `devtools.inspectedWindow.reload` command

    extension->reloadIgnoringCache(ignoreCache, std::nullopt, std::nullopt, [](Inspector::ExtensionVoidResult&& result) {
        if (!result)
            RELEASE_LOG_ERROR(Extensions, "Inspector could not reload page (%{public}@)", (NSString *)extensionErrorToString(result.error()));
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
