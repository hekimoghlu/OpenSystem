/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
#import "LegacyCustomProtocolManagerClient.h"

#import "CacheStoragePolicy.h"
#import "LegacyCustomProtocolManagerProxy.h"
#import <WebCore/ResourceError.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/ResourceResponse.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/SpanCocoa.h>

@interface WKCustomProtocolLoader : NSObject <NSURLConnectionDelegate> {
@private
    WeakPtr<WebKit::LegacyCustomProtocolManagerProxy> _customProtocolManagerProxy;
    Markable<WebKit::LegacyCustomProtocolID> _customProtocolID;
    NSURLCacheStoragePolicy _storagePolicy;
    RetainPtr<NSURLConnection> _urlConnection;
}
- (id)initWithLegacyCustomProtocolManagerProxy:(WebKit::LegacyCustomProtocolManagerProxy&)customProtocolManagerProxy customProtocolID:(WebKit::LegacyCustomProtocolID)customProtocolID request:(NSURLRequest *)request;
- (void)cancel;
@end

@implementation WKCustomProtocolLoader

- (id)initWithLegacyCustomProtocolManagerProxy:(WebKit::LegacyCustomProtocolManagerProxy&)customProtocolManagerProxy customProtocolID:(WebKit::LegacyCustomProtocolID)customProtocolID request:(NSURLRequest *)request
{
    self = [super init];
    if (!self)
        return nil;

    ASSERT(request);
    _customProtocolManagerProxy = customProtocolManagerProxy;
    _customProtocolID = customProtocolID;
    _storagePolicy = NSURLCacheStorageNotAllowed;
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    _urlConnection = adoptNS([[NSURLConnection alloc] initWithRequest:request delegate:self startImmediately:NO]);
    [_urlConnection scheduleInRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
    [_urlConnection start];
ALLOW_DEPRECATED_DECLARATIONS_END

    return self;
}

- (void)dealloc
{
    [_urlConnection cancel];
    [super dealloc];
}

- (void)cancel
{
    ASSERT(_customProtocolManagerProxy);
    _customProtocolManagerProxy = nullptr;
    [_urlConnection cancel];
}

- (void)connection:(NSURLConnection *)connection didFailWithError:(NSError *)error
{
    RefPtr customProtocolManagerProxy = _customProtocolManagerProxy.get();
    if (!customProtocolManagerProxy)
        return;

    WebCore::ResourceError coreError(error);
    customProtocolManagerProxy->didFailWithError(*_customProtocolID, coreError);
    customProtocolManagerProxy->stopLoading(*_customProtocolID);
}

- (NSCachedURLResponse *)connection:(NSURLConnection *)connection willCacheResponse:(NSCachedURLResponse *)cachedResponse
{
    ASSERT(_storagePolicy == NSURLCacheStorageNotAllowed);
    _storagePolicy = [cachedResponse storagePolicy];
    return cachedResponse;
}

- (void)connection:(NSURLConnection *)connection didReceiveResponse:(NSURLResponse *)response
{
    RefPtr customProtocolManagerProxy = _customProtocolManagerProxy.get();
    if (!customProtocolManagerProxy)
        return;

    WebCore::ResourceResponse coreResponse(response);
    customProtocolManagerProxy->didReceiveResponse(*_customProtocolID, coreResponse, WebKit::toCacheStoragePolicy(_storagePolicy));
}

- (void)connection:(NSURLConnection *)connection didReceiveData:(NSData *)data
{
    RefPtr customProtocolManagerProxy = _customProtocolManagerProxy.get();
    if (!customProtocolManagerProxy)
        return;

    customProtocolManagerProxy->didLoadData(*_customProtocolID, span(data));
}

- (NSURLRequest *)connection:(NSURLConnection *)connection willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse
{
    RefPtr customProtocolManagerProxy = _customProtocolManagerProxy.get();
    if (!customProtocolManagerProxy)
        return nil;

    if (redirectResponse) {
        customProtocolManagerProxy->wasRedirectedToRequest(*_customProtocolID, request, redirectResponse);
        return nil;
    }
    return request;
}

- (void)connectionDidFinishLoading:(NSURLConnection *)connection
{
    RefPtr customProtocolManagerProxy = _customProtocolManagerProxy.get();
    if (!customProtocolManagerProxy)
        return;

    customProtocolManagerProxy->didFinishLoading(*_customProtocolID);
    customProtocolManagerProxy->stopLoading(*_customProtocolID);
}

@end

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LegacyCustomProtocolManagerClient);

using namespace WebCore;

void LegacyCustomProtocolManagerClient::startLoading(LegacyCustomProtocolManagerProxy& manager, WebKit::LegacyCustomProtocolID customProtocolID, const ResourceRequest& coreRequest)
{
    NSURLRequest *request = coreRequest.nsURLRequest(HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
    if (!request)
        return;

    auto loader = adoptNS([[WKCustomProtocolLoader alloc] initWithLegacyCustomProtocolManagerProxy:manager customProtocolID:customProtocolID request:request]);
    ASSERT(loader);
    ASSERT(!m_loaderMap.contains(customProtocolID));
    m_loaderMap.add(customProtocolID, WTFMove(loader));
}

void LegacyCustomProtocolManagerClient::stopLoading(LegacyCustomProtocolManagerProxy&, WebKit::LegacyCustomProtocolID customProtocolID)
{
    m_loaderMap.remove(customProtocolID);
}

void LegacyCustomProtocolManagerClient::invalidate(LegacyCustomProtocolManagerProxy&)
{
    while (!m_loaderMap.isEmpty()) {
        auto loader = m_loaderMap.take(m_loaderMap.begin()->key);
        ASSERT(loader);
        [loader cancel];
    }
}

} // namespace WebKit
