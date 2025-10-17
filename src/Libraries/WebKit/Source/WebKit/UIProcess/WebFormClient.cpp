/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#include "WebFormClient.h"

#include "APIDictionary.h"
#include "APIString.h"
#include "WKAPICast.h"
#include "WebFormSubmissionListenerProxy.h"
#include "WebFrameProxy.h"
#include "WebPageProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebFormClient);

WebFormClient::WebFormClient(const WKPageFormClientBase* wkClient)
{
    initialize(wkClient);
}

void WebFormClient::willSubmitForm(WebPageProxy& page, WebFrameProxy& frame, WebFrameProxy& sourceFrame, const Vector<std::pair<String, String>>& textFieldValues, API::Object* userData, CompletionHandler<void()>&& completionHandler)
{
    if (!m_client.willSubmitForm) {
        completionHandler();
        return;
    }

    API::Dictionary::MapType map;
    for (size_t i = 0; i < textFieldValues.size(); ++i)
        map.set(textFieldValues[i].first, API::String::create(textFieldValues[i].second));
    auto textFieldsMap = API::Dictionary::create(WTFMove(map));
    auto listener = WebFormSubmissionListenerProxy::create(WTFMove(completionHandler));
    m_client.willSubmitForm(toAPI(&page), toAPI(&frame), toAPI(&sourceFrame), toAPI(textFieldsMap.ptr()), toAPI(userData), toAPI(listener.ptr()), m_client.base.clientInfo);
}

} // namespace WebKit
