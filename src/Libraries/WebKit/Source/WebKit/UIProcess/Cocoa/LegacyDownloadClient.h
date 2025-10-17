/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#pragma once

#import "APIDownloadClient.h"
#import "WKFoundation.h"
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@protocol _WKDownloadDelegate;

namespace WebCore {
class ResourceError;
class ResourceResponse;
}

namespace WebKit {

class LegacyDownloadClient final : public API::DownloadClient {
    WTF_MAKE_TZONE_ALLOCATED(LegacyDownloadClient);
public:
    explicit LegacyDownloadClient(id <_WKDownloadDelegate>);
    
private:
    // From API::DownloadClient
    void legacyDidStart(DownloadProxy&) final;
    void didReceiveResponse(DownloadProxy&, const WebCore::ResourceResponse&);
    void didReceiveData(DownloadProxy&, uint64_t, uint64_t, uint64_t) final;
    void decideDestinationWithSuggestedFilename(DownloadProxy&, const WebCore::ResourceResponse&, const String& suggestedFilename, CompletionHandler<void(AllowOverwrite, String)>&&) final;
    void didFinish(DownloadProxy&) final;
    void didFail(DownloadProxy&, const WebCore::ResourceError&, API::Data*) final;
    void legacyDidCancel(DownloadProxy&) final;
    void willSendRequest(DownloadProxy&, WebCore::ResourceRequest&&, const WebCore::ResourceResponse&, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) final;
    void didReceiveAuthenticationChallenge(DownloadProxy&, AuthenticationChallengeProxy&) final;
    void didCreateDestination(DownloadProxy&, const String&) final;
    void processDidCrash(DownloadProxy&) final;

    WeakObjCPtr<id <_WKDownloadDelegate>> m_delegate;

    struct {
        bool downloadDidStart : 1;            
        bool downloadDidReceiveResponse : 1;
        bool downloadDidReceiveData : 1;
        bool downloadDidWriteDataTotalBytesWrittenTotalBytesExpectedToWrite : 1;
        bool downloadDecideDestinationWithSuggestedFilenameAllowOverwrite : 1;
        bool downloadDecideDestinationWithSuggestedFilenameCompletionHandler : 1;
        bool downloadDidFinish : 1;
        bool downloadDidFail : 1;
        bool downloadDidCancel : 1;
        bool downloadDidReceiveServerRedirectToURL : 1;
        bool downloadDidReceiveAuthenticationChallengeCompletionHandler : 1;
        bool downloadShouldDecodeSourceDataOfMIMEType : 1;
        bool downloadDidCreateDestination : 1;
        bool downloadProcessDidCrash : 1;
    } m_delegateMethods;
};

} // namespace WebKit
