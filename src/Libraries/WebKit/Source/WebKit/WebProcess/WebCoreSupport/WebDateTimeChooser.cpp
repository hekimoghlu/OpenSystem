/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#include "WebDateTimeChooser.h"

#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/DateTimeChooserClient.h>
#include <WebCore/DateTimeChooserParameters.h>

namespace WebKit {

WebDateTimeChooser::WebDateTimeChooser(WebPage& page, WebCore::DateTimeChooserClient& client)
    : m_client(client)
    , m_page(page)
{
}

WebDateTimeChooser::~WebDateTimeChooser() = default;

void WebDateTimeChooser::didChooseDate(StringView date)
{
    m_client->didChooseValue(date);
}

void WebDateTimeChooser::didEndChooser()
{
    m_client->didEndChooser();
}

void WebDateTimeChooser::endChooser()
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::EndDateTimePicker(), page->identifier());
}

void WebDateTimeChooser::showChooser(const WebCore::DateTimeChooserParameters& params)
{
    RefPtr page  = m_page.get();
    if (!page)
        return;

    page->setActiveDateTimeChooser(*this);
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebPageProxy::ShowDateTimePicker(params), page->identifier());
}

} // namespace WebKit
