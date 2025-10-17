/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include "APINavigationResponse.h"

#include "APIFrameInfo.h"
#include "APINavigation.h"
#include "WebFrameProxy.h"

namespace API {

NavigationResponse::NavigationResponse(API::FrameInfo& frame, const WebCore::ResourceRequest& request, const WebCore::ResourceResponse& response, bool canShowMIMEType, const WTF::String& downloadAttribute, Navigation* navigation)
    : m_frame(frame)
    , m_request(request)
    , m_response(response)
    , m_canShowMIMEType(canShowMIMEType)
    , m_downloadAttribute(downloadAttribute)
    , m_navigation(navigation) { }

NavigationResponse::~NavigationResponse() = default;

FrameInfo* NavigationResponse::navigationInitiatingFrame()
{
    if (m_sourceFrame)
        return m_sourceFrame.get();
    if (!m_navigation)
        return nullptr;
    auto& frameInfo = m_navigation->originatingFrameInfo();
    RefPtr frame = WebKit::WebFrameProxy::webFrame(frameInfo.frameID);
    m_sourceFrame = FrameInfo::create(FrameInfoData { frameInfo }, frame ? frame->page() : nullptr);
    return m_sourceFrame.get();
}

}
