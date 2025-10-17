/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "FrameLoadRequest.h"

#include "Document.h"
#include "LocalFrame.h"
#include "SecurityOrigin.h"

namespace WebCore {

FrameLoadRequest::FrameLoadRequest(Ref<Document>&& requester, SecurityOrigin& requesterSecurityOrigin, ResourceRequest&& resourceRequest, const AtomString& frameName, InitiatedByMainFrame initiatedByMainFrame, const AtomString& downloadAttribute)
    : m_requester { WTFMove(requester) }
    , m_requesterSecurityOrigin { requesterSecurityOrigin }
    , m_resourceRequest { WTFMove(resourceRequest) }
    , m_frameName { frameName }
    , m_downloadAttribute { downloadAttribute }
    , m_initiatedByMainFrame { initiatedByMainFrame }
{
}

FrameLoadRequest::FrameLoadRequest(LocalFrame& frame, const ResourceRequest& resourceRequest, const SubstituteData& substituteData)
    : m_requester { *frame.document() }
    , m_requesterSecurityOrigin { frame.document()->securityOrigin() }
    , m_resourceRequest { resourceRequest }
    , m_substituteData { substituteData }
{
}

FrameLoadRequest::~FrameLoadRequest() = default;

FrameLoadRequest::FrameLoadRequest(FrameLoadRequest&&) = default;
FrameLoadRequest& FrameLoadRequest::operator=(FrameLoadRequest&&) = default;

Document& FrameLoadRequest::requester()
{
    return m_requester.get();
}

Ref<Document> FrameLoadRequest::protectedRequester() const
{
    return m_requester;
}

const SecurityOrigin& FrameLoadRequest::requesterSecurityOrigin() const
{
    return m_requesterSecurityOrigin.get();
}

Ref<SecurityOrigin> FrameLoadRequest::protectedRequesterSecurityOrigin() const
{
    return m_requesterSecurityOrigin;
}

} // namespace WebCore
