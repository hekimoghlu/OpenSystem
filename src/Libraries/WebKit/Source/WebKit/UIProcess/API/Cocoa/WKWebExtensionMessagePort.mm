/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#import "WKWebExtensionMessagePortInternal.h"

#import "WebExtensionMessagePort.h"
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>

NSErrorDomain const WKWebExtensionMessagePortErrorDomain = @"WKWebExtensionMessagePortErrorDomain";

@implementation WKWebExtensionMessagePort

#if ENABLE(WK_WEB_EXTENSIONS)

WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(WKWebExtensionMessagePort, WebExtensionMessagePort, _webExtensionMessagePort);

- (BOOL)isEqual:(id)object
{
    if (self == object)
        return YES;

    auto *other = dynamic_objc_cast<WKWebExtensionMessagePort>(object);
    if (!other)
        return NO;

    return *_webExtensionMessagePort == *other->_webExtensionMessagePort;
}

- (NSString *)applicationIdentifier
{
    if (auto& applicationIdentifier = _webExtensionMessagePort->applicationIdentifier(); !applicationIdentifier.isNull())
        return applicationIdentifier;
    return nil;
}

- (BOOL)isDisconnected
{
    return self._protectedWebExtensionMessagePort->isDisconnected();
}

- (void)sendMessage:(id)message completionHandler:(void (^)(NSError *))completionHandler
{
    if (!completionHandler)
        completionHandler = ^(NSError *) { };

    self._protectedWebExtensionMessagePort->sendMessage(message, [completionHandler = makeBlockPtr(completionHandler)](WebKit::WebExtensionMessagePort::Error error) {
        if (error) {
            completionHandler(toAPI(error.value()));
            return;
        }

        completionHandler(nil);
    });
}

- (void)disconnect
{
    [self disconnectWithError:nil];
}

- (void)disconnectWithError:(NSError *)error
{
    self._protectedWebExtensionMessagePort->disconnect(WebKit::toWebExtensionMessagePortError(error));
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_webExtensionMessagePort;
}

- (WebKit::WebExtensionMessagePort&)_webExtensionMessagePort
{
    return *_webExtensionMessagePort;
}

- (Ref<WebKit::WebExtensionMessagePort>)_protectedWebExtensionMessagePort
{
    return *_webExtensionMessagePort;
}

#else // ENABLE(WK_WEB_EXTENSIONS)

- (NSString *)applicationIdentifier
{
    return nil;
}

- (BOOL)isDisconnected
{
    return NO;
}

- (void)sendMessage:(id)message completionHandler:(void (^)(NSError *))completionHandler
{
    completionHandler([NSError errorWithDomain:NSCocoaErrorDomain code:NSFeatureUnsupportedError userInfo:nil]);
}

- (void)disconnect
{
}

- (void)disconnectWithError:(NSError *)error
{
}

#endif // ENABLE(WK_WEB_EXTENSIONS)

@end
