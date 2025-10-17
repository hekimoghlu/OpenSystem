/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#import "WKWebProcessPlugInInternal.h"

#import "APIArray.h"
#import "WKBundle.h"
#import "WKBundleAPICast.h"
#import "WKRetainPtr.h"
#import "WKStringCF.h"
#import "WKWebProcessPlugInBrowserContextControllerInternal.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/RetainPtr.h>
#import <wtf/StdLibExtras.h>

@interface WKWebProcessPlugInController () {
    API::ObjectStorage<WebKit::InjectedBundle> _bundle;
    RetainPtr<id <WKWebProcessPlugIn>> _principalClassInstance;
}
@end

@implementation WKWebProcessPlugInController

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWebProcessPlugInController.class, self))
        return;

    _bundle->~InjectedBundle();

    [super dealloc];
}

static void didCreatePage(WKBundleRef bundle, WKBundlePageRef page, const void* clientInfo)
{
    auto plugInController = (__bridge WKWebProcessPlugInController *)clientInfo;
    id <WKWebProcessPlugIn> principalClassInstance = plugInController->_principalClassInstance.get();

    if ([principalClassInstance respondsToSelector:@selector(webProcessPlugIn:didCreateBrowserContextController:)])
        [principalClassInstance webProcessPlugIn:plugInController didCreateBrowserContextController:wrapper(*WebKit::toImpl(page))];
}

static void willDestroyPage(WKBundleRef bundle, WKBundlePageRef page, const void* clientInfo)
{
    auto plugInController = (__bridge WKWebProcessPlugInController *)clientInfo;
    id <WKWebProcessPlugIn> principalClassInstance = plugInController->_principalClassInstance.get();

    if ([principalClassInstance respondsToSelector:@selector(webProcessPlugIn:willDestroyBrowserContextController:)])
        [principalClassInstance webProcessPlugIn:plugInController willDestroyBrowserContextController:wrapper(*WebKit::toImpl(page))];
}

static void setUpBundleClient(WKWebProcessPlugInController *plugInController, WebKit::InjectedBundle& bundle)
{
    WKBundleClientV1 bundleClient;
    zeroBytes(bundleClient);

    bundleClient.base.version = 1;
    bundleClient.base.clientInfo = (__bridge void*)plugInController;
    bundleClient.didCreatePage = didCreatePage;
    bundleClient.willDestroyPage = willDestroyPage;

    WKBundleSetClient(toAPI(&bundle), &bundleClient.base);
}

- (void)_setPrincipalClassInstance:(id <WKWebProcessPlugIn>)principalClassInstance
{
    ASSERT(!_principalClassInstance);
    _principalClassInstance = principalClassInstance;

    setUpBundleClient(self, *_bundle);
}

- (id)parameters
{
    return _bundle->bundleParameters();
}

static Ref<API::Array> createWKArray(NSArray *array)
{
    NSUInteger count = [array count];
    Vector<RefPtr<API::Object>> strings;
    strings.reserveInitialCapacity(count);
    
    for (id entry in array) {
        if ([entry isKindOfClass:[NSString class]])
            strings.append(adoptRef(WebKit::toImpl(WKStringCreateWithCFString((__bridge CFStringRef)entry))));
    }
    
    return API::Array::create(WTFMove(strings));
}

- (void)extendClassesForParameterCoder:(NSArray *)classes
{
    auto classList = createWKArray(classes);
    _bundle->extendClassesForParameterCoder(classList.get());
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_bundle;
}

@end

@implementation WKWebProcessPlugInController (Private)

- (WKBundleRef)_bundleRef
{
    return toAPI(_bundle.get());
}

@end
