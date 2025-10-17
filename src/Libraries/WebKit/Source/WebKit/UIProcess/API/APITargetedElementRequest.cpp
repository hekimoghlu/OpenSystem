/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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
#include "APITargetedElementRequest.h"

#include "PageClient.h"
#include "WebPageProxy.h"

namespace API {

WebCore::FloatPoint TargetedElementRequest::point() const
{
    if (!std::holds_alternative<WebCore::FloatPoint>(m_request.data))
        return { };

    return std::get<WebCore::FloatPoint>(m_request.data);
}

void TargetedElementRequest::setPoint(WebCore::FloatPoint point)
{
    m_request.data = point;
}

void TargetedElementRequest::setSelectors(WebCore::TargetedElementSelectors&& selectors)
{
    m_request.data = WTFMove(selectors);
}

void TargetedElementRequest::setSearchText(WTF::String&& searchText)
{
    m_request.data = WTFMove(searchText);
}

WebCore::TargetedElementRequest TargetedElementRequest::makeRequest(const WebKit::WebPageProxy& page) const
{
    auto request = m_request;
    if (std::holds_alternative<WebCore::FloatPoint>(m_request.data)) {
        if (RefPtr pageClient = page.pageClient())
            request.data = pageClient->webViewToRootView(point());
    }
    return request;
}

} // namespace API
