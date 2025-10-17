/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#import "WKURLSessionTaskDelegate.h"

#import "AuthenticationChallengeDispositionCocoa.h"
#import "Connection.h"
#import "NetworkProcess.h"
#import "NetworkProcessProxyMessages.h"
#import "NetworkSessionCocoa.h"
#import <Foundation/NSURLSession.h>
#import <WebCore/AuthenticationChallenge.h>
#import <WebCore/Credential.h>
#import <WebCore/ResourceError.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/ResourceResponse.h>
#import <wtf/BlockPtr.h>
#import <wtf/SystemTracing.h>
#import <wtf/cocoa/SpanCocoa.h>

@implementation WKURLSessionTaskDelegate {
    Markable<WebKit::DataTaskIdentifier> _identifier;
    WeakPtr<WebKit::NetworkSessionCocoa> _session;
}

- (instancetype)initWithTask:(NSURLSessionTask *)task identifier:(WebKit::DataTaskIdentifier)identifier session:(WebKit::NetworkSessionCocoa&)session
{
    if (!(self = [super init]))
        return nil;

    WTFBeginSignpost(self, DataTask, "%{public}@ %{private}@", task.originalRequest.HTTPMethod, task.originalRequest.URL);
    _identifier = identifier;
    _session = WeakPtr { session };

    return self;
}

- (void)dealloc
{
    WTFEndSignpost(self, DataTask);
    [super dealloc];
}

- (IPC::Connection*)connection
{
    if (!_session)
        return nil;
    return _session->networkProcess().parentProcessConnection();
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didReceiveChallenge:(NSURLAuthenticationChallenge *)challenge completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition disposition, NSURLCredential *credential))completionHandler
{
    WTFEmitSignpost(self, DataTask, "received challenge");
    RefPtr connection = [self connection];
    if (!connection)
        return completionHandler(NSURLSessionAuthChallengeRejectProtectionSpace, nil);
    connection->sendWithAsyncReply(Messages::NetworkProcessProxy::DataTaskReceivedChallenge(*_identifier, challenge), [completionHandler = makeBlockPtr(completionHandler)](WebKit::AuthenticationChallengeDisposition disposition, WebCore::Credential&& credential) {
        completionHandler(fromAuthenticationChallengeDisposition(disposition), credential.nsCredential());
    });
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task willPerformHTTPRedirection:(NSHTTPURLResponse *)response newRequest:(NSURLRequest *)request completionHandler:(void (^)(NSURLRequest *))completionHandler
{
    WTFEmitSignpost(self, DataTask, "redirect");
    RefPtr connection = [self connection];
    if (!connection)
        return completionHandler(nil);
    connection->sendWithAsyncReply(Messages::NetworkProcessProxy::DataTaskWillPerformHTTPRedirection(*_identifier, response, request), [completionHandler = makeBlockPtr(completionHandler), request = RetainPtr { request }] (bool allowed) {
        completionHandler(allowed ? request.get() : nil);
    });
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)dataTask didReceiveResponse:(NSURLResponse *)response completionHandler:(void (^)(NSURLSessionResponseDisposition disposition))completionHandler
{
    WTFEmitSignpost(self, DataTask, "received response headers");
    RefPtr connection = [self connection];
    if (!connection)
        return completionHandler(NSURLSessionResponseCancel);
    connection->sendWithAsyncReply(Messages::NetworkProcessProxy::DataTaskDidReceiveResponse(*_identifier, response), [completionHandler = makeBlockPtr(completionHandler)] (bool allowed) {
        completionHandler(allowed ? NSURLSessionResponseAllow : NSURLSessionResponseCancel);
    });
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)dataTask didReceiveData:(NSData *)data
{
    WTFEmitSignpost(self, DataTask, "received %zu bytes", static_cast<size_t>(data.length));
    RefPtr connection = [self connection];
    if (!connection)
        return;
    connection->send(Messages::NetworkProcessProxy::DataTaskDidReceiveData(*_identifier, span(data)), 0);
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didCompleteWithError:(NSError *)error
{
    WTFEmitSignpost(self, DataTask, "completed with error: %d", !!error);
    RefPtr connection = [self connection];
    if (!connection)
        return;
    connection->send(Messages::NetworkProcessProxy::DataTaskDidCompleteWithError(*_identifier, error), 0);
    if (_session)
        _session->removeDataTask(*_identifier);
}

@end
