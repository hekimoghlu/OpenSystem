/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#import "WebFrameNetworkingContext.h"

#import "NetworkSession.h"
#import "WebErrors.h"
#import "WebPage.h"
#import "WebProcess.h"
#import "WebsiteDataStoreParameters.h"
#import <WebCore/FrameLoader.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameLoaderClient.h>
#import <WebCore/NetworkStorageSession.h>
#import <WebCore/Page.h>
#import <WebCore/ResourceError.h>
#import <WebCore/Settings.h>
#import <wtf/ProcessPrivilege.h>

namespace WebKit {
using namespace WebCore;

bool WebFrameNetworkingContext::localFileContentSniffingEnabled() const
{
    return frame() && frame()->settings().localFileContentSniffingEnabled();
}

SchedulePairHashSet* WebFrameNetworkingContext::scheduledRunLoopPairs() const
{
    if (!frame() || !frame()->page())
        return nullptr;
    return frame()->page()->scheduledRunLoopPairs();
}

RetainPtr<CFDataRef> WebFrameNetworkingContext::sourceApplicationAuditData() const
{
    return WebProcess::singleton().sourceApplicationAuditData();
}

String WebFrameNetworkingContext::sourceApplicationIdentifier() const
{
    return WebProcess::singleton().uiProcessBundleIdentifier();
}

ResourceError WebFrameNetworkingContext::blockedError(const ResourceRequest& request) const
{
    return WebKit::blockedError(request);
}

WebLocalFrameLoaderClient* WebFrameNetworkingContext::webFrameLoaderClient() const
{
    if (!frame())
        return nullptr;

    return toWebLocalFrameLoaderClient(frame()->loader().client());
}

}
