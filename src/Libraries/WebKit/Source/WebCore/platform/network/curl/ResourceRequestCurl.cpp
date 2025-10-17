/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include "ResourceRequest.h"

#if USE(CURL)

namespace WebCore {

void ResourceRequest::updateFromDelegatePreservingOldProperties(const ResourceRequest& delegateProvidedRequest)
{
    // These are things we don't want willSendRequest delegate to mutate or reset.
    ResourceLoadPriority oldPriority = priority();
    RefPtr<FormData> oldHTTPBody = httpBody();
    bool isHiddenFromInspector = hiddenFromInspector();
    auto oldRequester = requester();
    auto oldInitiatorIdentifier = initiatorIdentifier();
    auto oldInspectorInitiatorNodeIdentifier = inspectorInitiatorNodeIdentifier();

    *this = delegateProvidedRequest;

    setPriority(oldPriority);
    setHTTPBody(WTFMove(oldHTTPBody));
    setHiddenFromInspector(isHiddenFromInspector);
    setRequester(oldRequester);
    setInitiatorIdentifier(oldInitiatorIdentifier);
    if (oldInspectorInitiatorNodeIdentifier)
        setInspectorInitiatorNodeIdentifier(*oldInspectorInitiatorNodeIdentifier);
}

ResourceRequest ResourceRequest::fromResourceRequestData(ResourceRequestBase::RequestData&& requestData)
{
    return ResourceRequest(WTFMove(requestData));
}

ResourceRequestBase::RequestData ResourceRequest::getRequestDataToSerialize() const
{
    return m_requestData;
}

}

#endif
