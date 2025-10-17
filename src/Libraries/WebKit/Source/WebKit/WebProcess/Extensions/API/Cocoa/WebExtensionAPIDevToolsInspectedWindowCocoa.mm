/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#import "WebExtensionAPIDevToolsInspectedWindow.h"

#if ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)

#import "APISerializedScriptValue.h"
#import "CocoaHelpers.h"
#import "JSWebExtensionWrapper.h"
#import "MessageSenderInlines.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionTabIdentifier.h"
#import "WebExtensionUtilities.h"
#import "WebProcess.h"

static NSString * const frameURLKey = @"frameURL";
static NSString * const ignoreCacheKey = @"ignoreCache";

static NSString * const isExceptionKey = @"isException";
static NSString * const valueKey = @"value";

namespace WebKit {

void WebExtensionAPIDevToolsInspectedWindow::eval(WebPageProxyIdentifier webPageProxyIdentifier, NSString *expression, NSDictionary *options, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/inspectedWindow/eval

    static NSDictionary<NSString *, id> *types = @{
        frameURLKey: NSString.class,
    };

    if (!validateDictionary(options, @"options", nil, types, outExceptionString))
        return;

    // FIXME: <https://webkit.org/b/269349> Implement `contextSecurityOrigin` and `useContentScriptContext` options for `devtools.inspectedWindow.eval` command

    std::optional<WTF::URL> frameURL;
    if (NSString *url = options[frameURLKey]) {
        frameURL = URL { url };

        if (!frameURL.value().isValid()) {
            *outExceptionString = toErrorString(nullString(), frameURLKey, @"'%@' is not a valid URL", url);
            return;
        }
    }

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DevToolsInspectedWindowEval(webPageProxyIdentifier, expression, frameURL), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<Expected<std::span<const uint8_t>, WebCore::ExceptionDetails>, WebExtensionError>&& result) mutable {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        auto *undefinedValue = [JSValue valueWithUndefinedInContext:[JSContext contextWithJSGlobalContextRef:callback->globalContext()]];

        if (!result.value()) {
            // If an error occurred, element 0 will be undefined, and element 1 will contain an object giving details about the error.
            callback->call(@[ undefinedValue, @{ isExceptionKey: @YES, valueKey: result.value().error().message } ]);
            return;
        }

        Ref serializedValue = API::SerializedScriptValue::createFromWireBytes(result.value().value());
        id scriptResult = API::SerializedScriptValue::deserialize(serializedValue->internalRepresentation());

        // If no error occurred, element 0 will contain the result of evaluating the expression, and element 1 will be undefined.
        callback->call(@[ scriptResult ?: undefinedValue, undefinedValue ]);
    }, extensionContext().identifier());
}

void WebExtensionAPIDevToolsInspectedWindow::reload(WebPageProxyIdentifier webPageProxyIdentifier, NSDictionary *options, NSString **outExceptionString)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/inspectedWindow/reload

    static NSDictionary<NSString *, id> *types = @{
        ignoreCacheKey: @YES.class,
    };

    if (!validateDictionary(options, @"options", nil, types, outExceptionString))
        return;

    // FIXME: <https://webkit.org/b/222328> Implement `userAgent` and `injectedScript` options for `devtools.inspectedWindow.reload` command

    std::optional<bool> ignoreCache;
    if (NSNumber *value = options[ignoreCacheKey])
        ignoreCache = value.boolValue;

    WebProcess::singleton().send(Messages::WebExtensionContext::DevToolsInspectedWindowReload(webPageProxyIdentifier, ignoreCache), extensionContext().identifier());
}

double WebExtensionAPIDevToolsInspectedWindow::tabId(WebPage& page)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/inspectedWindow/tabId

    auto result = extensionContext().tabIdentifier(page);
    return toWebAPI(result ? result.value() : WebExtensionTabConstants::NoneIdentifier);
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
