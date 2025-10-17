/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#import "config.h"
#import "_WKInspectorExtensionInternal.h"

#if ENABLE(INSPECTOR_EXTENSIONS)

#import "APISerializedScriptValue.h"
#import "InspectorExtensionDelegate.h"
#import "InspectorExtensionTypes.h"
#import "WKError.h"
#import "WKWebViewInternal.h"
#import <WebCore/ExceptionDetails.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/BlockPtr.h>
#import <wtf/URL.h>

@implementation _WKInspectorExtension

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKInspectorExtension.class, self))
        return;

    _extension->API::InspectorExtension::~InspectorExtension();

    [super dealloc];
}

- (API::Object&)_apiObject
{
    return *_extension;
}

- (id <_WKInspectorExtensionDelegate>)delegate
{
    return _delegate->delegate().autorelease();
}

- (void)setDelegate:(id<_WKInspectorExtensionDelegate>)delegate
{
    if (!delegate && !_delegate)
        return;

    _delegate = makeUnique<WebKit::InspectorExtensionDelegate>(self, delegate);
}

// MARK: API

- (void)createTabWithName:(NSString *)tabName tabIconURL:(NSURL *)tabIconURL sourceURL:(NSURL *)sourceURL completionHandler:(void(^)(NSError *, NSString *))completionHandler
{
    _extension->createTab(tabName, { tabIconURL.baseURL, tabIconURL.relativeString }, { tabIconURL.baseURL, sourceURL.relativeString }, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(completionHandler)] (Expected<Inspector::ExtensionTabID, Inspector::ExtensionError> result) mutable {
        if (!result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.error())}], nil);
            return;
        }

        capturedBlock(nil, result.value());
    });
}

- (void)evaluateScript:(NSString *)scriptSource frameURL:(NSURL *)frameURL contextSecurityOrigin:(NSURL *)contextSecurityOrigin useContentScriptContext:(BOOL)useContentScriptContext completionHandler:(void(^)(NSError *, id))completionHandler
{
    std::optional<URL> optionalFrameURL = frameURL ? std::make_optional(URL(frameURL)) : std::nullopt;
    std::optional<URL> optionalContextSecurityOrigin = contextSecurityOrigin ? std::make_optional(URL(contextSecurityOrigin)) : std::nullopt;
    _extension->evaluateScript(scriptSource, optionalFrameURL, optionalContextSecurityOrigin, useContentScriptContext, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(WTFMove(completionHandler))] (Inspector::ExtensionEvaluationResult&& result) mutable {
        if (!result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.error())}], nil);
            return;
        }
        
        auto valueOrException = result.value();
        if (!valueOrException) {
            capturedBlock(nsErrorFromExceptionDetails(valueOrException.error()).get(), nil);
            return;
        }

        id body = API::SerializedScriptValue::deserialize(valueOrException.value()->internalRepresentation());
        capturedBlock(nil, body);
    });
}

- (void)evaluateScript:(NSString *)scriptSource inTabWithIdentifier:(NSString *)extensionTabIdentifier completionHandler:(void(^)(NSError *, id))completionHandler
{
    _extension->evaluateScriptInExtensionTab(extensionTabIdentifier, scriptSource, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(WTFMove(completionHandler))] (Inspector::ExtensionEvaluationResult&& result) mutable {
        if (!result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.error()) }], nil);
            return;
        }
        
        auto valueOrException = result.value();
        if (!valueOrException) {
            capturedBlock(nsErrorFromExceptionDetails(valueOrException.error()).get(), nil);
            return;
        }

        id body = API::SerializedScriptValue::deserialize(valueOrException.value()->internalRepresentation());
        capturedBlock(nil, body);
    });
}

- (void)navigateToURL:(NSURL *)url inTabWithIdentifier:(NSString *)extensionTabIdentifier completionHandler:(void(^)(NSError *))completionHandler
{
    _extension->navigateTab(extensionTabIdentifier, url, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(WTFMove(completionHandler))] (const std::optional<Inspector::ExtensionError> result) mutable {
        if (result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.value()) }]);
            return;
        }

        capturedBlock(nil);
    });
}

- (void)reloadIgnoringCache:(BOOL)ignoreCache userAgent:(NSString *)userAgent injectedScript:(NSString *)injectedScript completionHandler:(void(^)(NSError *))completionHandler
{
    std::optional<String> optionalUserAgent = userAgent ? std::make_optional(String(userAgent)) : std::nullopt;
    std::optional<String> optionalInjectedScript = injectedScript ? std::make_optional(String(injectedScript)) : std::nullopt;
    _extension->reloadIgnoringCache(ignoreCache, optionalUserAgent, optionalInjectedScript, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(WTFMove(completionHandler))] (Inspector::ExtensionVoidResult&& result) mutable {
        if (!result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.error()) }]);
            return;
        }

        capturedBlock(nil);
    });
}

// MARK: Properties.

- (NSString *)extensionID
{
    return _extension->identifier();
}

@end

#endif // ENABLE(INSPECTOR_EXTENSIONS)
