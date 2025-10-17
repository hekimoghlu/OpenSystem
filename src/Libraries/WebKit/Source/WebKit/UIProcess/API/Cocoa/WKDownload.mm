/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#import "WKDownloadInternal.h"

#import "APIDownloadClient.h"
#import "CompletionHandlerCallChecker.h"
#import "DownloadProxy.h"
#import "WKDownloadDelegate.h"
#import "WKDownloadDelegatePrivate.h"
#import "WKFrameInfoInternal.h"
#import "WKNSData.h"
#import "WKNSURLAuthenticationChallenge.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import <Foundation/Foundation.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/WeakObjCPtr.h>

class DownloadClient final : public API::DownloadClient {
public:
    explicit DownloadClient(id<WKDownloadDelegatePrivate> delegate)
        : m_delegate(delegate)
        , m_respondsToWillPerformHTTPRedirection([delegate respondsToSelector:@selector(download:willPerformHTTPRedirection:newRequest:decisionHandler:)])
        , m_respondsToDidReceiveAuthenticationChallenge([delegate respondsToSelector:@selector(download:didReceiveAuthenticationChallenge:completionHandler:)])
        , m_respondsToDidFinish([m_delegate respondsToSelector:@selector(downloadDidFinish:)])
        , m_respondsToDidFailWithError([delegate respondsToSelector:@selector(download:didFailWithError:resumeData:)])
        , m_respondsToDecidePlaceholderPolicy([delegate respondsToSelector:@selector(_download:decidePlaceholderPolicy:)])
        , m_respondsToDecidePlaceholderPolicyAPI([delegate respondsToSelector:@selector(download:decidePlaceholderPolicy:)])
#if HAVE(MODERN_DOWNLOADPROGRESS)
        , m_respondsToDidReceivePlaceholderURL([delegate respondsToSelector:@selector(_download:didReceivePlaceholderURL:completionHandler:)])
        , m_respondsToDidReceivePlaceholderURLAPI([delegate respondsToSelector:@selector(download:didReceivePlaceholderURL:completionHandler:)])
        , m_respondsToDidReceiveFinalURL([delegate respondsToSelector:@selector(_download:didReceiveFinalURL:)])
        , m_respondsToDidReceiveFinalURLAPI([delegate respondsToSelector:@selector(download:didReceiveFinalURL:)])
#endif

    {
        ASSERT([delegate respondsToSelector:@selector(download:decideDestinationUsingResponse:suggestedFilename:completionHandler:)]);
    }

private:
    void willSendRequest(WebKit::DownloadProxy& download, WebCore::ResourceRequest&& request, const WebCore::ResourceResponse& response, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler) final
    {
        if (!m_delegate || !m_respondsToWillPerformHTTPRedirection)
            return completionHandler(WTFMove(request));

        RetainPtr<NSURLRequest> nsRequest = request.nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
        [m_delegate download:wrapper(download) willPerformHTTPRedirection:static_cast<NSHTTPURLResponse *>(response.nsURLResponse()) newRequest:nsRequest.get() decisionHandler:makeBlockPtr([request = WTFMove(request), completionHandler = WTFMove(completionHandler), checker = WebKit::CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(download:willPerformHTTPRedirection:newRequest:decisionHandler:))](WKDownloadRedirectPolicy policy) mutable {
            if (checker->completionHandlerHasBeenCalled())
                return;
            checker->didCallCompletionHandler();
            switch (policy) {
            case WKDownloadRedirectPolicyCancel:
                return completionHandler({ });
            case WKDownloadRedirectPolicyAllow:
                return completionHandler(WTFMove(request));
            default:
                [NSException raise:NSInvalidArgumentException format:@"Invalid WKDownloadRedirectPolicy (%ld)", (long)policy];
            }
        }).get()];
    }

    void didReceiveAuthenticationChallenge(WebKit::DownloadProxy& download, WebKit::AuthenticationChallengeProxy& challenge) final
    {
        if (!m_delegate || !m_respondsToDidReceiveAuthenticationChallenge)
            return challenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue);

        [m_delegate download:wrapper(download) didReceiveAuthenticationChallenge:wrapper(challenge) completionHandler:makeBlockPtr([challenge = Ref { challenge }, checker = WebKit::CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(download:didReceiveAuthenticationChallenge:completionHandler:))] (NSURLSessionAuthChallengeDisposition disposition, NSURLCredential *credential) mutable {
            if (checker->completionHandlerHasBeenCalled())
                return;
            checker->didCallCompletionHandler();
            switch (disposition) {
            case NSURLSessionAuthChallengeUseCredential:
                challenge->listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::UseCredential, credential ? WebCore::Credential(credential) : WebCore::Credential());
                break;
            case NSURLSessionAuthChallengePerformDefaultHandling:
                challenge->listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::PerformDefaultHandling);
                break;
            case NSURLSessionAuthChallengeCancelAuthenticationChallenge:
                challenge->listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::Cancel);
                break;
            case NSURLSessionAuthChallengeRejectProtectionSpace:
                challenge->listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue);
                break;
            default:
                [NSException raise:NSInvalidArgumentException format:@"Invalid NSURLSessionAuthChallengeDisposition (%ld)", (long)disposition];
            }
        }).get()];
    }

    void decideDestinationWithSuggestedFilename(WebKit::DownloadProxy& download, const WebCore::ResourceResponse& response, const WTF::String& suggestedFilename, CompletionHandler<void(WebKit::AllowOverwrite, WTF::String)>&& completionHandler) final
    {
        if (!m_delegate)
            return completionHandler(WebKit::AllowOverwrite::No, { });

        [m_delegate download:wrapper(download) decideDestinationUsingResponse:response.nsURLResponse() suggestedFilename:suggestedFilename completionHandler:makeBlockPtr([download = Ref { download }, completionHandler = WTFMove(completionHandler), checker = WebKit::CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(download:decideDestinationUsingResponse:suggestedFilename:completionHandler:))] (NSURL *destination) mutable {
            if (checker->completionHandlerHasBeenCalled())
                return;
            checker->didCallCompletionHandler();

            if (!destination)
                return completionHandler(WebKit::AllowOverwrite::No, { });
            
            if (!destination.isFileURL) {
                completionHandler(WebKit::AllowOverwrite::No, { });
                [NSException raise:NSInvalidArgumentException format:@"destination must be a file URL"];
                return;
            }

            NSFileManager *fileManager = [NSFileManager defaultManager];
            if (![fileManager fileExistsAtPath:[destination URLByDeletingLastPathComponent].path])
                return completionHandler(WebKit::AllowOverwrite::No, { });
            if ([fileManager fileExistsAtPath:destination.path])
                return completionHandler(WebKit::AllowOverwrite::No, { });

            wrapper(download.get()).progress.fileURL = destination;

            completionHandler(WebKit::AllowOverwrite::No, destination.path);
        }).get()];
    }

    void decidePlaceholderPolicy(WebKit::DownloadProxy& download, CompletionHandler<void(WebKit::UseDownloadPlaceholder, const WTF::URL&)>&& completionHandler)
    {
        if (!m_respondsToDecidePlaceholderPolicy && !m_respondsToDecidePlaceholderPolicyAPI) {
            completionHandler(WebKit::UseDownloadPlaceholder::No, { });
            return;
        }
        if (m_respondsToDecidePlaceholderPolicy) {
            [m_delegate _download:wrapper(download) decidePlaceholderPolicy:makeBlockPtr([completionHandler = WTFMove(completionHandler)] (_WKPlaceholderPolicy policy, NSURL *alternatePlaceholderURL) mutable {
                switch (policy) {
                case _WKPlaceholderPolicyDisable: {
                    completionHandler(WebKit::UseDownloadPlaceholder::No, alternatePlaceholderURL);
                    break;
                }
                case _WKPlaceholderPolicyEnable: {
                    completionHandler(WebKit::UseDownloadPlaceholder::Yes, alternatePlaceholderURL);
                    break;
                }
                default:
                    [NSException raise:NSInvalidArgumentException format:@"Invalid WKPlaceholderPolicy (%ld)", (long)policy];
                }
            }).get()];
        } else {
            [m_delegate download:wrapper(download) decidePlaceholderPolicy:makeBlockPtr([completionHandler = WTFMove(completionHandler)] (WKDownloadPlaceholderPolicy policy, NSURL *alternatePlaceholderURL) mutable {
                switch (policy) {
                case WKDownloadPlaceholderPolicyDisable: {
                    completionHandler(WebKit::UseDownloadPlaceholder::No, alternatePlaceholderURL);
                    break;
                }
                case WKDownloadPlaceholderPolicyEnable: {
                    completionHandler(WebKit::UseDownloadPlaceholder::Yes, alternatePlaceholderURL);
                    break;
                }
                default:
                    [NSException raise:NSInvalidArgumentException format:@"Invalid WKDownloadPlaceholderPolicy (%ld)", (long)policy];
                }
            }).get()];
        }
    }

    void didReceiveData(WebKit::DownloadProxy& download, uint64_t, uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite) final
    {
        NSProgress *progress = wrapper(download).progress;
        progress.totalUnitCount = totalBytesExpectedToWrite;
        progress.completedUnitCount = totalBytesWritten;
    }

    void didFinish(WebKit::DownloadProxy& download) final
    {
        if (!m_delegate || !m_respondsToDidFinish)
            return;

        [m_delegate downloadDidFinish:wrapper(download)];
    }

    void didFail(WebKit::DownloadProxy& download, const WebCore::ResourceError& error, API::Data* resumeData) final
    {
        if (!m_delegate || !m_respondsToDidFailWithError)
            return;

        [m_delegate download:wrapper(download) didFailWithError:error.nsError() resumeData:wrapper(resumeData)];
    }

    void processDidCrash(WebKit::DownloadProxy& download) final
    {
        if (!m_delegate || !m_respondsToDidFailWithError)
            return;

        [m_delegate download:wrapper(download) didFailWithError:[NSError errorWithDomain:NSURLErrorDomain code:NSURLErrorNetworkConnectionLost userInfo:nil] resumeData:nil];
    }

#if HAVE(MODERN_DOWNLOADPROGRESS)
    void didReceivePlaceholderURL(WebKit::DownloadProxy& download, const WTF::URL& url, std::span<const uint8_t> bookmarkData, CompletionHandler<void()>&& completionHandler) final
    {
        if (!m_delegate || (!m_respondsToDidReceivePlaceholderURL && !m_respondsToDidReceivePlaceholderURLAPI)) {
            completionHandler();
            return;
        }

        BOOL bookmarkDataIsStale = NO;
        NSError *bookmarkResolvingError;
        RetainPtr data = toNSData(bookmarkData);
        RetainPtr urlFromBookmark = adoptNS([[NSURL alloc] initByResolvingBookmarkData:data.get() options:0 relativeToURL:nil bookmarkDataIsStale:&bookmarkDataIsStale error:&bookmarkResolvingError]);
        if (bookmarkResolvingError || bookmarkDataIsStale)
            RELEASE_LOG_ERROR(Network, "Failed to resolve URL from bookmark data");

        NSURL *placeholderURL = urlFromBookmark ? urlFromBookmark.get() : (NSURL *)url;

        if (m_respondsToDidReceivePlaceholderURL)
            [m_delegate _download:wrapper(download) didReceivePlaceholderURL:placeholderURL completionHandler:makeBlockPtr(WTFMove(completionHandler)).get()];
        else
            [m_delegate download:wrapper(download) didReceivePlaceholderURL:placeholderURL completionHandler:makeBlockPtr(WTFMove(completionHandler)).get()];
    }

    void didReceiveFinalURL(WebKit::DownloadProxy& download, const WTF::URL& url, std::span<const uint8_t> bookmarkData) final
    {
        if (!m_delegate || (!m_respondsToDidReceiveFinalURL && !m_respondsToDidReceiveFinalURLAPI))
            return;

        BOOL bookmarkDataIsStale = NO;
        NSError *bookmarkResolvingError;
        RetainPtr data = toNSData(bookmarkData);
        RetainPtr urlFromBookmark = adoptNS([[NSURL alloc] initByResolvingBookmarkData:data.get() options:0 relativeToURL:nil bookmarkDataIsStale:&bookmarkDataIsStale error:&bookmarkResolvingError]);
        if (bookmarkResolvingError || bookmarkDataIsStale)
            RELEASE_LOG_ERROR(Network, "Failed to resolve URL from bookmark data");

        NSURL *finalURL = urlFromBookmark.get() ?: (NSURL *)url;

        if (m_respondsToDidReceiveFinalURL)
            [m_delegate _download:wrapper(download) didReceiveFinalURL:finalURL];
        else
            [m_delegate download:wrapper(download) didReceiveFinalURL:finalURL];
    }
#endif

    WeakObjCPtr<id<WKDownloadDelegatePrivate>> m_delegate;

    bool m_respondsToWillPerformHTTPRedirection : 1;
    bool m_respondsToDidReceiveAuthenticationChallenge : 1;
    bool m_respondsToDidFinish : 1;
    bool m_respondsToDidFailWithError : 1;
    bool m_respondsToDecidePlaceholderPolicy : 1;
    bool m_respondsToDecidePlaceholderPolicyAPI : 1;
#if HAVE(MODERN_DOWNLOADPROGRESS)
    bool m_respondsToDidReceivePlaceholderURL : 1;
    bool m_respondsToDidReceivePlaceholderURLAPI : 1;
    bool m_respondsToDidReceiveFinalURL : 1;
    bool m_respondsToDidReceiveFinalURLAPI : 1;
#endif
};

@implementation WKDownload

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)cancel:(void (^)(NSData *resumeData))completionHandler
{
    _download->cancel([completionHandler = makeBlockPtr(completionHandler)] (auto* data) {
        if (completionHandler)
            completionHandler(wrapper(data));
    });
}

- (NSURLRequest *)originalRequest
{
    return _download->request().nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
}

- (WKWebView *)webView
{
    auto page = _download->originatingPage();
    return page ? page->cocoaView().autorelease() : nil;
}

- (BOOL)isUserInitiated
{
    return _download->wasUserInitiated();
}

- (WKFrameInfo *)originatingFrame
{
    return WebKit::wrapper(_download->frameInfo());
}

- (id <WKDownloadDelegate>)delegate
{
    return _delegate.get().get();
}

- (void)setDelegate:(id<WKDownloadDelegatePrivate>)delegate
{
    _delegate = delegate;
    _download->setClient(adoptRef(*new DownloadClient(delegate)));
}

#pragma mark NSProgressReporting protocol implementation

- (NSProgress *)progress
{
    NSProgress* downloadProgress = _download->progress();
    if (!downloadProgress) {
        constexpr auto indeterminateUnitCount = -1;
        downloadProgress = [NSProgress progressWithTotalUnitCount:indeterminateUnitCount];

        downloadProgress.kind = NSProgressKindFile;
        downloadProgress.fileOperationKind = NSProgressFileOperationKindDownloading;

        downloadProgress.cancellable = YES;
        downloadProgress.cancellationHandler = makeBlockPtr([weakSelf = WeakObjCPtr<WKDownload> { self }] () mutable {
            ensureOnMainRunLoop([weakSelf = WTFMove(weakSelf)] {
                [weakSelf cancel:nil];
            });
        }).get();

        _download->setProgress(downloadProgress);
    }
    return downloadProgress;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKDownload.class, self))
        return;
    _download->~DownloadProxy();
    [super dealloc];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_download;
}

@end
