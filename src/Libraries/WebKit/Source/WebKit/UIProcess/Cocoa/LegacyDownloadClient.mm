/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#import "LegacyDownloadClient.h"

#import "AuthenticationChallengeDisposition.h"
#import "AuthenticationChallengeProxy.h"
#import "AuthenticationDecisionListener.h"
#import "CompletionHandlerCallChecker.h"
#import "DownloadProxy.h"
#import "Logging.h"
#import "WKDownloadInternal.h"
#import "WKNSURLAuthenticationChallenge.h"
#import "WKNSURLExtras.h"
#import "WebCredential.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import "_WKDownloadDelegate.h"
#import "_WKDownloadInternal.h"
#import <WebCore/ResourceError.h>
#import <WebCore/ResourceResponse.h>
#import <wtf/BlockPtr.h>
#import <wtf/FileSystem.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

ALLOW_DEPRECATED_DECLARATIONS_BEGIN

WTF_MAKE_TZONE_ALLOCATED_IMPL(LegacyDownloadClient);

LegacyDownloadClient::LegacyDownloadClient(id <_WKDownloadDelegate> delegate)
    : m_delegate(delegate)
{
    m_delegateMethods.downloadDidStart = [delegate respondsToSelector:@selector(_downloadDidStart:)];
    m_delegateMethods.downloadDidReceiveResponse = [delegate respondsToSelector:@selector(_download:didReceiveResponse:)];
    m_delegateMethods.downloadDidReceiveData = [delegate respondsToSelector:@selector(_download:didReceiveData:)];
    m_delegateMethods.downloadDidWriteDataTotalBytesWrittenTotalBytesExpectedToWrite = [delegate respondsToSelector:@selector(_download:didWriteData:totalBytesWritten:totalBytesExpectedToWrite:)];
    m_delegateMethods.downloadDecideDestinationWithSuggestedFilenameAllowOverwrite = [delegate respondsToSelector:@selector(_download:decideDestinationWithSuggestedFilename:allowOverwrite:)];
    m_delegateMethods.downloadDecideDestinationWithSuggestedFilenameCompletionHandler = [delegate respondsToSelector:@selector(_download:decideDestinationWithSuggestedFilename:completionHandler:)];
    m_delegateMethods.downloadDidFinish = [delegate respondsToSelector:@selector(_downloadDidFinish:)];
    m_delegateMethods.downloadDidFail = [delegate respondsToSelector:@selector(_download:didFailWithError:)];
    m_delegateMethods.downloadDidCancel = [delegate respondsToSelector:@selector(_downloadDidCancel:)];
    m_delegateMethods.downloadDidReceiveServerRedirectToURL = [delegate respondsToSelector:@selector(_download:didReceiveServerRedirectToURL:)];
    m_delegateMethods.downloadDidReceiveAuthenticationChallengeCompletionHandler = [delegate respondsToSelector:@selector(_download:didReceiveAuthenticationChallenge:completionHandler:)];
    m_delegateMethods.downloadShouldDecodeSourceDataOfMIMEType = [delegate respondsToSelector:@selector(_download:shouldDecodeSourceDataOfMIMEType:)];
    m_delegateMethods.downloadDidCreateDestination = [delegate respondsToSelector:@selector(_download:didCreateDestination:)];
    m_delegateMethods.downloadProcessDidCrash = [delegate respondsToSelector:@selector(_downloadProcessDidCrash:)];
}

void LegacyDownloadClient::legacyDidStart(DownloadProxy& downloadProxy)
{
    if (m_delegateMethods.downloadDidStart)
        [m_delegate _downloadDidStart:[_WKDownload downloadWithDownload:wrapper(downloadProxy)]];
}

void LegacyDownloadClient::didReceiveResponse(DownloadProxy& downloadProxy, const WebCore::ResourceResponse& response)
{
    if (m_delegateMethods.downloadDidReceiveResponse)
        [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] didReceiveResponse:response.nsURLResponse()];
}

void LegacyDownloadClient::didReceiveData(DownloadProxy& downloadProxy, uint64_t bytesWritten, uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite)
{
    if (m_delegateMethods.downloadDidWriteDataTotalBytesWrittenTotalBytesExpectedToWrite)
        [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] didWriteData:bytesWritten totalBytesWritten:totalBytesWritten totalBytesExpectedToWrite:totalBytesExpectedToWrite];
    else if (m_delegateMethods.downloadDidReceiveData)
        [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] didReceiveData:bytesWritten];
}

void LegacyDownloadClient::didReceiveAuthenticationChallenge(DownloadProxy& downloadProxy, AuthenticationChallengeProxy& authenticationChallenge)
{
    // FIXME: System Preview needs code here.
    if (!m_delegateMethods.downloadDidReceiveAuthenticationChallengeCompletionHandler) {
        authenticationChallenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::PerformDefaultHandling);
        return;
    }

    [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] didReceiveAuthenticationChallenge:wrapper(authenticationChallenge) completionHandler:makeBlockPtr([authenticationChallenge = Ref { authenticationChallenge }, checker = CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(_download:didReceiveAuthenticationChallenge:completionHandler:))] (NSURLSessionAuthChallengeDisposition disposition, NSURLCredential *credential) {
        if (checker->completionHandlerHasBeenCalled())
            return;
        checker->didCallCompletionHandler();
        switch (disposition) {
        case NSURLSessionAuthChallengeUseCredential:
            authenticationChallenge->listener().completeChallenge(AuthenticationChallengeDisposition::UseCredential, credential ? WebCore::Credential(credential) : WebCore::Credential());
            break;
        case NSURLSessionAuthChallengePerformDefaultHandling:
            authenticationChallenge->listener().completeChallenge(AuthenticationChallengeDisposition::PerformDefaultHandling);
            break;
            
        case NSURLSessionAuthChallengeCancelAuthenticationChallenge:
            authenticationChallenge->listener().completeChallenge(AuthenticationChallengeDisposition::Cancel);
            break;
            
        case NSURLSessionAuthChallengeRejectProtectionSpace:
            authenticationChallenge->listener().completeChallenge(AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue);
            break;
            
        default:
            [NSException raise:NSInvalidArgumentException format:@"Invalid NSURLSessionAuthChallengeDisposition (%ld)", (long)disposition];
        }
    }).get()];
}

void LegacyDownloadClient::didCreateDestination(DownloadProxy& downloadProxy, const String& destination)
{
    if (m_delegateMethods.downloadDidCreateDestination)
        [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] didCreateDestination:destination];
}

void LegacyDownloadClient::processDidCrash(DownloadProxy& downloadProxy)
{
    if (m_delegateMethods.downloadProcessDidCrash)
        [m_delegate _downloadProcessDidCrash:[_WKDownload downloadWithDownload:wrapper(downloadProxy)]];
}

void LegacyDownloadClient::decideDestinationWithSuggestedFilename(DownloadProxy& downloadProxy, const WebCore::ResourceResponse& response, const String& filename, CompletionHandler<void(AllowOverwrite, String)>&& completionHandler)
{
    didReceiveResponse(downloadProxy, response);

    if (!m_delegateMethods.downloadDecideDestinationWithSuggestedFilenameAllowOverwrite && !m_delegateMethods.downloadDecideDestinationWithSuggestedFilenameCompletionHandler)
        return completionHandler(AllowOverwrite::No, { });

    if (m_delegateMethods.downloadDecideDestinationWithSuggestedFilenameAllowOverwrite) {
        BOOL allowOverwrite = NO;
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        NSString *destination = [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] decideDestinationWithSuggestedFilename:filename allowOverwrite:&allowOverwrite];
ALLOW_DEPRECATED_DECLARATIONS_END
        completionHandler(allowOverwrite ? AllowOverwrite::Yes : AllowOverwrite::No, destination);
    } else {
        [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] decideDestinationWithSuggestedFilename:filename completionHandler:makeBlockPtr([checker = CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(_download:decideDestinationWithSuggestedFilename:completionHandler:)), completionHandler = WTFMove(completionHandler)] (BOOL allowOverwrite, NSString *destination) mutable {
            if (checker->completionHandlerHasBeenCalled())
                return;
            checker->didCallCompletionHandler();
            completionHandler(allowOverwrite ? AllowOverwrite::Yes : AllowOverwrite::No, destination);
        }).get()];
    }
}

void LegacyDownloadClient::didFinish(DownloadProxy& downloadProxy)
{
    if (m_delegateMethods.downloadDidFinish)
        [m_delegate _downloadDidFinish:[_WKDownload downloadWithDownload:wrapper(downloadProxy)]];
}

void LegacyDownloadClient::didFail(DownloadProxy& downloadProxy, const WebCore::ResourceError& error, API::Data*)
{
    if (m_delegateMethods.downloadDidFail)
        [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] didFailWithError:error.nsError()];
}

void LegacyDownloadClient::legacyDidCancel(DownloadProxy& downloadProxy)
{
    if (m_delegateMethods.downloadDidCancel)
        [m_delegate _downloadDidCancel:[_WKDownload downloadWithDownload:wrapper(downloadProxy)]];
}

void LegacyDownloadClient::willSendRequest(DownloadProxy& downloadProxy, WebCore::ResourceRequest&& request, const WebCore::ResourceResponse&, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    if (m_delegateMethods.downloadDidReceiveServerRedirectToURL)
        [m_delegate _download:[_WKDownload downloadWithDownload:wrapper(downloadProxy)] didReceiveServerRedirectToURL:request.url()];

    completionHandler(WTFMove(request));
}

ALLOW_DEPRECATED_DECLARATIONS_END

} // namespace WebKit
