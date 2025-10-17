/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#import <WebKitLegacy/WebDownload.h>

#import "NetworkStorageSessionMap.h"
#import <Foundation/NSURLAuthenticationChallenge.h>
#import <WebCore/AuthenticationMac.h>
#import <WebCore/Credential.h>
#import <WebCore/CredentialStorage.h>
#import <WebCore/NetworkStorageSession.h>
#import <WebCore/ProtectionSpace.h>
#import <WebKitLegacy/WebPanelAuthenticationHandler.h>
#import <pal/spi/cocoa/NSURLDownloadSPI.h>
#import <wtf/Assertions.h>
#import <wtf/MainThread.h>
#import <wtf/WorkQueue.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#import <wtf/spi/darwin/dyldSPI.h>

static bool shouldCallOnNetworkThread()
{
#if PLATFORM(MAC)
    static bool isOldEpsonSoftwareUpdater = WTF::MacApplication::isEpsonSoftwareUpdater() && !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DownloadDelegatesCalledOnTheMainThread);
    return isOldEpsonSoftwareUpdater;
#else
    return false;
#endif
}

static void callOnDelegateThread(Function<void()>&& function)
{
    if (shouldCallOnNetworkThread())
        function();
    callOnMainThread(WTFMove(function));
}

template<typename Callable>
static void callOnDelegateThreadAndWait(Callable&& work)
{
    if (shouldCallOnNetworkThread() || isMainThread())
        work();
    else {
        WorkQueue::main().dispatchSync([work = std::forward<Callable>(work)]() mutable {
            work();
        });
    }
}

using namespace WebCore;

@interface WebDownloadInternal : NSObject <NSURLDownloadDelegate> {
    RetainPtr<id> realDelegate;
}
- (void)setRealDelegate:(id)realDelegate;
@end

@implementation WebDownloadInternal

- (void)dealloc
{
    [super dealloc];
}

- (void)setRealDelegate:(id)rd
{
    realDelegate = rd;
}

- (BOOL)respondsToSelector:(SEL)selector
{
    if (selector == @selector(downloadDidBegin:) ||
        selector == @selector(download:willSendRequest:redirectResponse:) ||
        selector == @selector(download:didReceiveResponse:) ||
        selector == @selector(download:didReceiveDataOfLength:) ||
        selector == @selector(download:shouldDecodeSourceDataOfMIMEType:) ||
        selector == @selector(download:decideDestinationWithSuggestedFilename:) ||
        selector == @selector(download:didCreateDestination:) ||
        selector == @selector(downloadDidFinish:) ||
        selector == @selector(download:didFailWithError:)) {
        return [realDelegate respondsToSelector:selector];
    }

    return [super respondsToSelector:selector];
}

- (void)downloadDidBegin:(NSURLDownload *)download
{
    callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download)] {
        [realDelegate downloadDidBegin:download.get()];
    });
}

- (NSURLRequest *)download:(NSURLDownload *)download willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse
{
    RetainPtr<NSURLRequest> returnValue;
    auto work = [&] {
        ASSERT(isMainThread());
        returnValue = [realDelegate download:download willSendRequest:request redirectResponse:redirectResponse];
    };
    callOnDelegateThreadAndWait(WTFMove(work));
    return returnValue.autorelease();
}

- (void)download:(NSURLDownload *)download didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
#if !PLATFORM(IOS_FAMILY)
    // Try previously stored credential first.
    if (![challenge previousFailureCount]) {
        NSURLCredential *credential = NetworkStorageSessionMap::defaultStorageSession().credentialStorage().get(emptyString(), ProtectionSpace([challenge protectionSpace])).nsCredential();
        if (credential) {
            [[challenge sender] useCredential:credential forAuthenticationChallenge:challenge];
            return;
        }
    }

    if ([realDelegate respondsToSelector:@selector(download:didReceiveAuthenticationChallenge:)]) {
        callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download), challenge = retainPtr(challenge)] {
            [realDelegate download:download.get() didReceiveAuthenticationChallenge:challenge.get()];
        });
    } else {
        callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download), challenge = retainPtr(challenge)] {
            NSWindow *window = nil;
            if ([realDelegate respondsToSelector:@selector(downloadWindowForAuthenticationSheet:)])
                window = [realDelegate downloadWindowForAuthenticationSheet:(WebDownload *)download.get()];

            [[WebPanelAuthenticationHandler sharedHandler] startAuthentication:challenge.get() window:window];
        });
    }
#endif
}

- (void)download:(NSURLDownload *)download didReceiveResponse:(NSURLResponse *)response
{
    callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download), response = retainPtr(response)] {
        [realDelegate download:download.get() didReceiveResponse:response.get()];
    });
}

- (void)download:(NSURLDownload *)download didReceiveDataOfLength:(NSUInteger)length
{
    callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download), length] {
        [realDelegate download:download.get() didReceiveDataOfLength:length];
    });
}

- (BOOL)download:(NSURLDownload *)download shouldDecodeSourceDataOfMIMEType:(NSString *)encodingType
{
    BOOL returnValue = NO;
    auto work = [&] {
        returnValue = [realDelegate download:download shouldDecodeSourceDataOfMIMEType:encodingType];
    };
    callOnDelegateThreadAndWait(WTFMove(work));
    return returnValue;
}

- (void)download:(NSURLDownload *)download decideDestinationWithSuggestedFilename:(NSString *)filename
{
    callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download), filename = retainPtr(filename)] {
        [realDelegate download:download.get() decideDestinationWithSuggestedFilename:filename.get()];
    });
}

- (void)download:(NSURLDownload *)download didCreateDestination:(NSString *)path
{
    callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download), path = retainPtr(path)] {
        [realDelegate download:download.get() didCreateDestination:path.get()];
    });
}

- (void)downloadDidFinish:(NSURLDownload *)download
{
    callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download)] {
        [realDelegate downloadDidFinish:download.get()];
    });
}

- (void)download:(NSURLDownload *)download didFailWithError:(NSError *)error
{
    callOnDelegateThread([realDelegate = realDelegate, download = retainPtr(download), error = retainPtr(error)] {
        [realDelegate download:download.get() didFailWithError:error.get()];
    });
}

@end

@implementation WebDownload

- (void)_setRealDelegate:(id)delegate
{
    if (_webInternal == nil) {
        _webInternal = [[WebDownloadInternal alloc] init];
        [_webInternal setRealDelegate:delegate];
    } else {
        ASSERT(_webInternal == delegate);
    }
}

- (id)init
{
    self = [super init];
    if (self == nil)
        return nil;

    // _webInternal can be set up before init by _setRealDelegate
    if (_webInternal == nil)
        _webInternal = [[WebDownloadInternal alloc] init];

    return self;
}

- (void)dealloc
{
    [_webInternal release];
    [super dealloc];
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (id)initWithRequest:(NSURLRequest *)request delegate:(id<NSURLDownloadDelegate>)delegate
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    [self _setRealDelegate:delegate];
    return [super initWithRequest:request delegate:_webInternal];
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (id)_initWithLoadingConnection:(NSURLConnection *)connection
                         request:(NSURLRequest *)request
                        response:(NSURLResponse *)response
                        delegate:(id)delegate
                           proxy:(NSURLConnectionDelegateProxy *)proxy
IGNORE_WARNINGS_END
{
    [self _setRealDelegate:delegate];
    return [super _initWithLoadingConnection:connection request:request response:response delegate:_webInternal proxy:proxy];
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (id)_initWithRequest:(NSURLRequest *)request
              delegate:(id)delegate
             directory:(NSString *)directory
IGNORE_WARNINGS_END
{
    [self _setRealDelegate:delegate];
    return [super _initWithRequest:request delegate:_webInternal directory:directory];
}

@end
