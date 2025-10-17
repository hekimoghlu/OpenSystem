/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#import "_WKInspectorConfigurationInternal.h"

#import "WKProcessPoolInternal.h"
#import <WebKit/WKWebViewConfigurationPrivate.h>
#import "WebURLSchemeHandlerCocoa.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKInspectorConfiguration

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<API::InspectorConfiguration>(self);
    
    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKInspectorConfiguration.class, self))
        return;
    _configuration->API::InspectorConfiguration::~InspectorConfiguration();
    [super dealloc];
}

- (API::Object&)_apiObject
{
    return *_configuration;
}

- (void)setURLSchemeHandler:(id <WKURLSchemeHandler>)urlSchemeHandler forURLScheme:(NSString *)urlScheme
{
    _configuration->addURLSchemeHandler(WebKit::WebURLSchemeHandlerCocoa::create(urlSchemeHandler), urlScheme);
}

- (void)setProcessPool:(WKProcessPool *)processPool
{
    _configuration->setProcessPool(processPool ? processPool->_processPool.get() : nullptr);
}

- (WKProcessPool *)processPool
{
    return wrapper(_configuration->processPool());
}

- (void)applyToWebViewConfiguration:(WKWebViewConfiguration *)configuration
{
    for (auto pair : _configuration->urlSchemeHandlers()) {
        auto& handler = static_cast<WebKit::WebURLSchemeHandlerCocoa&>(pair.first.get());
        [configuration setURLSchemeHandler:handler.apiHandler() forURLScheme:pair.second];
    }

    if (auto* processPool = self.processPool)
        [configuration setProcessPool:processPool];

    if (auto* groupIdentifier = self.groupIdentifier)
        [configuration _setGroupIdentifier:groupIdentifier];
}

- (id)copyWithZone:(NSZone *)zone
{
    _WKInspectorConfiguration *configuration = [(_WKInspectorConfiguration *)[[self class] allocWithZone:zone] init];

    for (auto pair : _configuration->urlSchemeHandlers()) {
        auto& handler = static_cast<WebKit::WebURLSchemeHandlerCocoa&>(pair.first.get());
        [configuration setURLSchemeHandler:handler.apiHandler() forURLScheme:pair.second];
    }

    if (auto* processPool = self.processPool)
        [configuration setProcessPool:processPool];

    if (auto* groupIdentifier = self.groupIdentifier)
        [configuration setGroupIdentifier:groupIdentifier];

    return configuration;
}

@end
