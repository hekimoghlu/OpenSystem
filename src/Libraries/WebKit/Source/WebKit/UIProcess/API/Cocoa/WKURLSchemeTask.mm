/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#import "WKURLSchemeTask.h"

#import "WKFrameInfoInternal.h"
#import "WKURLSchemeTaskInternal.h"
#import "WebURLSchemeHandler.h"
#import "WebURLSchemeTask.h"
#import <WebCore/ResourceError.h>
#import <WebCore/ResourceResponse.h>
#import <WebCore/SharedBuffer.h>
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>
#import <wtf/MainThread.h>

static WebKit::WebURLSchemeTask::ExceptionType getExceptionTypeFromMainRunLoop(Function<WebKit::WebURLSchemeTask::ExceptionType ()>&& function)
{
    WebKit::WebURLSchemeTask::ExceptionType exceptionType;
    callOnMainRunLoopAndWait([function = WTFMove(function), &exceptionType] {
        exceptionType = function();
    });

    return exceptionType;
}

static void raiseExceptionIfNecessary(WebKit::WebURLSchemeTask::ExceptionType exceptionType)
{
    switch (exceptionType) {
    case WebKit::WebURLSchemeTask::ExceptionType::None:
        return;
    case WebKit::WebURLSchemeTask::ExceptionType::TaskAlreadyStopped:
        [NSException raise:NSInternalInconsistencyException format:@"This task has already been stopped"];
        break;
    case WebKit::WebURLSchemeTask::ExceptionType::CompleteAlreadyCalled:
        [NSException raise:NSInternalInconsistencyException format:@"[WKURLSchemeTask taskDidCompleteWithError:] has already been called for this task"];
        break;
    case WebKit::WebURLSchemeTask::ExceptionType::DataAlreadySent:
        [NSException raise:NSInternalInconsistencyException format:@"[WKURLSchemeTask taskDidReceiveData:] has already been called for this task"];
        break;
    case WebKit::WebURLSchemeTask::ExceptionType::NoResponseSent:
        [NSException raise:NSInternalInconsistencyException format:@"No response has been sent for this task"];
        break;
    case WebKit::WebURLSchemeTask::ExceptionType::RedirectAfterResponse:
        [NSException raise:NSInternalInconsistencyException format:@"No redirects are allowed after the response"];
        break;
    case WebKit::WebURLSchemeTask::ExceptionType::WaitingForRedirectCompletionHandler:
        [NSException raise:NSInternalInconsistencyException format:@"No callbacks are allowed while waiting for the redirection completion handler to be invoked"];
        break;
    }
}

@implementation WKURLSchemeTaskImpl

- (instancetype)init
{
    RELEASE_ASSERT_NOT_REACHED();
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKURLSchemeTaskImpl.class, self))
        return;
    _urlSchemeTask->WebURLSchemeTask::~WebURLSchemeTask();
    [super dealloc];
}

- (NSURLRequest *)request
{
    return _urlSchemeTask->nsRequest();
}

- (BOOL)_requestOnlyIfCached
{
    return _urlSchemeTask->nsRequest().cachePolicy == NSURLRequestReturnCacheDataDontLoad;
}

- (void)_willPerformRedirection:(NSURLResponse *)response newRequest:(NSURLRequest *)request completionHandler:(void (^)(NSURLRequest *))completionHandler
{
    auto function = [protectedSelf = retainPtr(self), self, protectedResponse = retainPtr(response), response, protectedRequest = retainPtr(request), request, handler = makeBlockPtr(completionHandler)] () mutable {
        return _urlSchemeTask->willPerformRedirection(response, request, [handler = WTFMove(handler)] (WebCore::ResourceRequest&& actualNewRequest) {
            handler.get()(actualNewRequest.nsURLRequest(WebCore::HTTPBodyUpdatePolicy::UpdateHTTPBody));
        });
    };

    auto result = getExceptionTypeFromMainRunLoop(WTFMove(function));
    raiseExceptionIfNecessary(result);
}

- (void)didReceiveResponse:(NSURLResponse *)response
{
    auto function = [protectedSelf = retainPtr(self), self, protectedResponse = retainPtr(response), response] {
        return _urlSchemeTask->didReceiveResponse(response);
    };

    auto result = getExceptionTypeFromMainRunLoop(WTFMove(function));
    raiseExceptionIfNecessary(result);
}

- (void)didReceiveData:(NSData *)data
{
    auto function = [protectedSelf = retainPtr(self), self, protectedData = retainPtr(data), data] () mutable {
        return _urlSchemeTask->didReceiveData(WebCore::SharedBuffer::create(data));
    };

    auto result = getExceptionTypeFromMainRunLoop(WTFMove(function));
    raiseExceptionIfNecessary(result);
}

- (void)didFinish
{
    auto function = [protectedSelf = retainPtr(self), self] {
        return _urlSchemeTask->didComplete({ });
    };

    auto result = getExceptionTypeFromMainRunLoop(WTFMove(function));
    raiseExceptionIfNecessary(result);
}

- (void)didFailWithError:(NSError *)error
{
    auto function = [protectedSelf = retainPtr(self), self, protectedError = retainPtr(error), error] {
        return _urlSchemeTask->didComplete(error);
    };

    auto result = getExceptionTypeFromMainRunLoop(WTFMove(function));
    raiseExceptionIfNecessary(result);
}

- (void)_didPerformRedirection:(NSURLResponse *)response newRequest:(NSURLRequest *)request
{
    auto function = [protectedSelf = retainPtr(self), self, protectedResponse = retainPtr(response), response, protectedRequest = retainPtr(request), request] {
        return _urlSchemeTask->didPerformRedirection(response, request);
    };

    auto result = getExceptionTypeFromMainRunLoop(WTFMove(function));
    raiseExceptionIfNecessary(result);
}

- (WKFrameInfo *)_frame
{
    return wrapper(_urlSchemeTask->frameInfo());
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_urlSchemeTask;
}

@end
