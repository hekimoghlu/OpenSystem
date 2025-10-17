/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#import "LegacyCustomProtocolManager.h"

#import "CacheStoragePolicy.h"
#import "LegacyCustomProtocolManagerMessages.h"
#import "NetworkProcess.h"
#import <Foundation/NSURLSession.h>
#import <WebCore/ResourceError.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/ResourceResponse.h>
#import <pal/spi/cocoa/NSURLConnectionSPI.h>
#import <pal/text/TextEncoding.h>
#import <wtf/URL.h>
#import <wtf/cocoa/SpanCocoa.h>

using namespace WebKit;

static RefPtr<NetworkProcess>& firstNetworkProcess()
{
    static NeverDestroyed<RefPtr<NetworkProcess>> networkProcess;
    return networkProcess.get();
}

static RefPtr<NetworkProcess> protectedFirstNetworkProcess()
{
    return firstNetworkProcess();
}

void LegacyCustomProtocolManager::networkProcessCreated(NetworkProcess& networkProcess)
{
    auto hasRegisteredSchemes = [] (auto* legacyCustomProtocolManager) {
        if (!legacyCustomProtocolManager)
            return false;
        Locker locker { legacyCustomProtocolManager->m_registeredSchemesLock };
        return !legacyCustomProtocolManager->m_registeredSchemes.isEmpty();
    };

    RELEASE_ASSERT(!firstNetworkProcess() || !hasRegisteredSchemes(RefPtr { protectedFirstNetworkProcess()->supplement<LegacyCustomProtocolManager>() }.get()));
    firstNetworkProcess() = &networkProcess;
}

@interface WKCustomProtocol : NSURLProtocol {
@private
    Markable<LegacyCustomProtocolID> _customProtocolID;
    RetainPtr<CFRunLoopRef> _initializationRunLoop;
}
@property (nonatomic, readonly) Markable<LegacyCustomProtocolID> customProtocolID;
@property (nonatomic, readonly) CFRunLoopRef initializationRunLoop;
@end

@implementation WKCustomProtocol

@synthesize customProtocolID = _customProtocolID;

+ (BOOL)canInitWithRequest:(NSURLRequest *)request
{
    // FIXME: This code runs in a dispatch queue so we can't ref NetworkProcess here.
    if (SUPPRESS_UNCOUNTED_LOCAL auto* customProtocolManager = protectedFirstNetworkProcess()->supplement<LegacyCustomProtocolManager>())
        SUPPRESS_UNCOUNTED_ARG return customProtocolManager->supportsScheme([[[request URL] scheme] lowercaseString]);
    return NO;
}

+ (NSURLRequest *)canonicalRequestForRequest:(NSURLRequest *)request
{
    return request;
}

+ (BOOL)requestIsCacheEquivalent:(NSURLRequest *)a toRequest:(NSURLRequest *)b
{
    return NO;
}

- (id)initWithRequest:(NSURLRequest *)request cachedResponse:(NSCachedURLResponse *)cachedResponse client:(id<NSURLProtocolClient>)client
{
    self = [super initWithRequest:request cachedResponse:cachedResponse client:client];
    if (!self)
        return nil;

    if (RefPtr customProtocolManager = protectedFirstNetworkProcess()->supplement<LegacyCustomProtocolManager>())
        _customProtocolID = customProtocolManager->addCustomProtocol(self);
    _initializationRunLoop = CFRunLoopGetCurrent();

    return self;
}

- (CFRunLoopRef)initializationRunLoop
{
    return _initializationRunLoop.get();
}

- (void)startLoading
{
    ensureOnMainRunLoop([customProtocolID = *self.customProtocolID, request = retainPtr([self request])] {
        if (RefPtr customProtocolManager = protectedFirstNetworkProcess()->supplement<LegacyCustomProtocolManager>())
            customProtocolManager->startLoading(customProtocolID, request.get());
    });
}

- (void)stopLoading
{
    ensureOnMainRunLoop([customProtocolID = *self.customProtocolID] {
        if (RefPtr customProtocolManager = protectedFirstNetworkProcess()->supplement<LegacyCustomProtocolManager>()) {
            customProtocolManager->stopLoading(customProtocolID);
            customProtocolManager->removeCustomProtocol(customProtocolID);
        }
    });
}

@end

namespace WebKit {

void LegacyCustomProtocolManager::registerProtocolClass()
{
}

void LegacyCustomProtocolManager::registerProtocolClass(NSURLSessionConfiguration *configuration)
{
    configuration.protocolClasses = @[[WKCustomProtocol class]];
}

void LegacyCustomProtocolManager::registerScheme(const String& scheme)
{
    ASSERT(!scheme.isNull());
    Locker locker { m_registeredSchemesLock };
    m_registeredSchemes.add(scheme);
}

void LegacyCustomProtocolManager::unregisterScheme(const String& scheme)
{
    ASSERT(!scheme.isNull());
    Locker locker { m_registeredSchemesLock };
    m_registeredSchemes.remove(scheme);
}

bool LegacyCustomProtocolManager::supportsScheme(const String& scheme)
{
    if (scheme.isNull())
        return false;

    Locker locker { m_registeredSchemesLock };
    return m_registeredSchemes.contains(scheme);
}

static inline void dispatchOnInitializationRunLoop(WKCustomProtocol* protocol, void (^block)())
{
    CFRunLoopRef runloop = protocol.initializationRunLoop;
    CFRunLoopPerformBlock(runloop, kCFRunLoopDefaultMode, block);
    CFRunLoopWakeUp(runloop);
}

void LegacyCustomProtocolManager::didFailWithError(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceError& error)
{
    RetainPtr<WKCustomProtocol> protocol = protocolForID(customProtocolID);
    if (!protocol)
        return;

    RetainPtr<NSError> nsError = error.nsError();

    dispatchOnInitializationRunLoop(protocol.get(), ^ {
        [[protocol client] URLProtocol:protocol.get() didFailWithError:nsError.get()];
    });

    removeCustomProtocol(customProtocolID);
}

void LegacyCustomProtocolManager::didLoadData(LegacyCustomProtocolID customProtocolID, std::span<const uint8_t> data)
{
    RetainPtr<WKCustomProtocol> protocol = protocolForID(customProtocolID);
    if (!protocol)
        return;

    RetainPtr nsData = toNSData(data);

    dispatchOnInitializationRunLoop(protocol.get(), ^ {
        [[protocol client] URLProtocol:protocol.get() didLoadData:nsData.get()];
    });
}

void LegacyCustomProtocolManager::didReceiveResponse(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceResponse& response, CacheStoragePolicy cacheStoragePolicy)
{
    RetainPtr<WKCustomProtocol> protocol = protocolForID(customProtocolID);
    if (!protocol)
        return;

    RetainPtr<NSURLResponse> nsResponse = response.nsURLResponse();

    dispatchOnInitializationRunLoop(protocol.get(), ^ {
        [[protocol client] URLProtocol:protocol.get() didReceiveResponse:nsResponse.get() cacheStoragePolicy:toNSURLCacheStoragePolicy(cacheStoragePolicy)];
    });
}

void LegacyCustomProtocolManager::didFinishLoading(LegacyCustomProtocolID customProtocolID)
{
    RetainPtr<WKCustomProtocol> protocol = protocolForID(customProtocolID);
    if (!protocol)
        return;

    dispatchOnInitializationRunLoop(protocol.get(), ^ {
        [[protocol client] URLProtocolDidFinishLoading:protocol.get()];
    });

    removeCustomProtocol(customProtocolID);
}

void LegacyCustomProtocolManager::wasRedirectedToRequest(LegacyCustomProtocolID customProtocolID, const WebCore::ResourceRequest& request, const WebCore::ResourceResponse& redirectResponse)
{
    RetainPtr<WKCustomProtocol> protocol = protocolForID(customProtocolID);
    if (!protocol)
        return;

    RetainPtr<NSURLRequest> nsRequest = request.nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
    RetainPtr<NSURLResponse> nsRedirectResponse = redirectResponse.nsURLResponse();

    dispatchOnInitializationRunLoop(protocol.get(), [protocol, nsRequest, nsRedirectResponse]() {
        [[protocol client] URLProtocol:protocol.get() wasRedirectedToRequest:nsRequest.get() redirectResponse:nsRedirectResponse.get()];
    });
}

RetainPtr<WKCustomProtocol> LegacyCustomProtocolManager::protocolForID(LegacyCustomProtocolID customProtocolID)
{
    Locker locker { m_customProtocolMapLock };

    CustomProtocolMap::const_iterator it = m_customProtocolMap.find(customProtocolID);
    if (it == m_customProtocolMap.end())
        return nil;
    
    ASSERT(it->value);
    return it->value;
}

} // namespace WebKit
