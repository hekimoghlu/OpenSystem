/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#include "config.h"
#include "WKFramePolicyListener.h"

#include "APIWebsitePolicies.h"
#include "WKAPICast.h"
#include "WebFramePolicyListenerProxy.h"
#include "WebFrameProxy.h"
#include "WebProcessPool.h"
#include "WebsiteDataStore.h"
#include "WebsitePoliciesData.h"

using namespace WebKit;

WKTypeID WKFramePolicyListenerGetTypeID()
{
    return toAPI(WebFramePolicyListenerProxy::APIType);
}

void WKFramePolicyListenerUse(WKFramePolicyListenerRef policyListenerRef)
{
    toImpl(policyListenerRef)->use();
}

void WKFramePolicyListenerUseInNewProcess(WKFramePolicyListenerRef policyListenerRef)
{
    toImpl(policyListenerRef)->use(nullptr, ProcessSwapRequestedByClient::Yes);
}

static void useWithPolicies(WKFramePolicyListenerRef policyListenerRef, WKWebsitePoliciesRef websitePolicies, ProcessSwapRequestedByClient processSwapRequestedByClient)
{
    toImpl(policyListenerRef)->use(toImpl(websitePolicies), processSwapRequestedByClient);
}

void WKFramePolicyListenerUseWithPolicies(WKFramePolicyListenerRef policyListenerRef, WKWebsitePoliciesRef websitePolicies)
{
    useWithPolicies(policyListenerRef, websitePolicies, ProcessSwapRequestedByClient::No);
}

void WKFramePolicyListenerUseInNewProcessWithPolicies(WKFramePolicyListenerRef policyListenerRef, WKWebsitePoliciesRef websitePolicies)
{
    useWithPolicies(policyListenerRef, websitePolicies, ProcessSwapRequestedByClient::Yes);
}

void WKFramePolicyListenerDownload(WKFramePolicyListenerRef policyListenerRef)
{
    toImpl(policyListenerRef)->download();
}

void WKFramePolicyListenerIgnore(WKFramePolicyListenerRef policyListenerRef)
{
    toImpl(policyListenerRef)->ignore();
}
