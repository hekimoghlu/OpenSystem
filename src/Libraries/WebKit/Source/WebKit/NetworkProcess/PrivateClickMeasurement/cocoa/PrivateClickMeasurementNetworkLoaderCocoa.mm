/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
#import "PrivateClickMeasurementNetworkLoader.h"

#import "NetworkDataTaskCocoa.h"
#import <WebCore/HTTPHeaderValues.h>
#import <WebCore/MIMETypeRegistry.h>
#import <WebCore/UserAgent.h>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/cocoa/SpanCocoa.h>

static RetainPtr<SecTrustRef>& allowedLocalTestServerTrust()
{
    static NeverDestroyed<RetainPtr<SecTrustRef>> serverTrust;
    return serverTrust.get();
}

static bool trustsServerForLocalTests(NSURLAuthenticationChallenge *challenge)
{
    if (![challenge.protectionSpace.host isEqualToString:@"127.0.0.1"]
        || !allowedLocalTestServerTrust())
        return false;

    return WebCore::certificatesMatch(allowedLocalTestServerTrust().get(), challenge.protectionSpace.serverTrust);
}

@interface WKNetworkSessionDelegateAllowingOnlyNonRedirectedJSON : NSObject <NSURLSessionDataDelegate>
@end

@implementation WKNetworkSessionDelegateAllowingOnlyNonRedirectedJSON

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task willPerformHTTPRedirection:(NSHTTPURLResponse *)response newRequest:(NSURLRequest *)request completionHandler:(void (^)(NSURLRequest *))completionHandler
{
    completionHandler(nil);
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)dataTask didReceiveResponse:(NSURLResponse *)response completionHandler:(void (^)(NSURLSessionResponseDisposition disposition))completionHandler
{
    if (WebCore::MIMETypeRegistry::isSupportedJSONMIMEType(response.MIMEType))
        return completionHandler(NSURLSessionResponseAllow);
    completionHandler(NSURLSessionResponseCancel);
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didReceiveChallenge:(NSURLAuthenticationChallenge *)challenge completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition disposition, NSURLCredential *credential))completionHandler
{
    if ([challenge.protectionSpace.authenticationMethod isEqualToString:NSURLAuthenticationMethodServerTrust]
        && trustsServerForLocalTests(challenge))
        return completionHandler(NSURLSessionAuthChallengeUseCredential, [NSURLCredential credentialForTrust:challenge.protectionSpace.serverTrust]);
    completionHandler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
}

@end

namespace WebKit::PCM {

enum class LoadTaskIdentifierType { };
using LoadTaskIdentifier = ObjectIdentifier<LoadTaskIdentifierType>;
static HashMap<LoadTaskIdentifier, RetainPtr<NSURLSessionDataTask>>& taskMap()
{
    static NeverDestroyed<HashMap<LoadTaskIdentifier, RetainPtr<NSURLSessionDataTask>>> map;
    return map.get();
}

static NSURLSession *statelessSessionWithoutRedirects()
{
    static NeverDestroyed<RetainPtr<WKNetworkSessionDelegateAllowingOnlyNonRedirectedJSON>> delegate = adoptNS([WKNetworkSessionDelegateAllowingOnlyNonRedirectedJSON new]);
    static NeverDestroyed<RetainPtr<NSURLSession>> session = [&] {
        NSURLSessionConfiguration *configuration = [NSURLSessionConfiguration ephemeralSessionConfiguration];
        configuration.HTTPCookieAcceptPolicy = NSHTTPCookieAcceptPolicyNever;
        configuration.URLCredentialStorage = nil;
        configuration.URLCache = nil;
        configuration.HTTPCookieStorage = nil;
        configuration._shouldSkipPreferredClientCertificateLookup = YES;
        return [NSURLSession sessionWithConfiguration:configuration delegate:delegate.get().get() delegateQueue:[NSOperationQueue mainQueue]];
    }();
    return session.get().get();
}

void NetworkLoader::allowTLSCertificateChainForLocalPCMTesting(const WebCore::CertificateInfo& certificateInfo)
{
    allowedLocalTestServerTrust() = certificateInfo.trust();
}

void NetworkLoader::start(URL&& url, RefPtr<JSON::Object>&& jsonPayload, WebCore::PrivateClickMeasurement::PcmDataCarried pcmDataCarried, Callback&& callback)
{
    // Prevent contacting non-local servers when a test certificate chain is used for 127.0.0.1.
    // FIXME: Use a proxy server to have tests cover the reports sent to the destination, too.
    if (allowedLocalTestServerTrust() && url.host() != "127.0.0.1"_s)
        return callback({ }, { });

    auto request = adoptNS([[NSMutableURLRequest alloc] initWithURL:url]);
    [request setValue:WebCore::HTTPHeaderValues::maxAge0() forHTTPHeaderField:@"Cache-Control"];
    [request setValue:WebCore::standardUserAgentWithApplicationName({ }) forHTTPHeaderField:@"User-Agent"];
    if (jsonPayload) {
        request.get().HTTPMethod = @"POST";
        [request setValue:WebCore::HTTPHeaderValues::applicationJSONContentType() forHTTPHeaderField:@"Content-Type"];
        auto body = jsonPayload->toJSONString().utf8();
        request.get().HTTPBody = toNSData(body.span()).get();
    }

    setPCMDataCarriedOnRequest(pcmDataCarried, request.get());

    auto identifier = LoadTaskIdentifier::generate();
    NSURLSessionDataTask *task = [statelessSessionWithoutRedirects() dataTaskWithRequest:request.get() completionHandler:makeBlockPtr([callback = WTFMove(callback), identifier](NSData *data, NSURLResponse *response, NSError *error) mutable {
        taskMap().remove(identifier);
        if (error)
            return callback(error.localizedDescription, { });
        if (auto jsonValue = JSON::Value::parseJSON(String::fromUTF8(span(data))))
            return callback({ }, jsonValue->asObject());
        callback({ }, nullptr);
    }).get()];
    [task resume];
    taskMap().add(identifier, task);
}

} // namespace WebKit::PCM
