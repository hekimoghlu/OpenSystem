/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#include "WebOpenPanelResultListenerProxy.h"

#include "APIArray.h"
#include "APIString.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <wtf/URL.h>
#include <wtf/Vector.h>

namespace WebKit {
using namespace WebCore;

WebOpenPanelResultListenerProxy::WebOpenPanelResultListenerProxy(WebPageProxy* page, WebProcessProxy& process)
    : m_page(page)
    , m_process(process)
{
}

WebOpenPanelResultListenerProxy::~WebOpenPanelResultListenerProxy()
{
}

#if PLATFORM(IOS_FAMILY)
void WebOpenPanelResultListenerProxy::chooseFiles(const Vector<WTF::String>& filenames, const String& displayString, const API::Data* iconImageData)
{
    if (!m_page)
        return;

    m_page->didChooseFilesForOpenPanelWithDisplayStringAndIcon(filenames, displayString, iconImageData);
}
#endif

void WebOpenPanelResultListenerProxy::chooseFiles(const Vector<String>& filenames, const Vector<String>& allowedMIMETypes)
{
    if (!m_page)
        return;

    m_page->didChooseFilesForOpenPanel(filenames, allowedMIMETypes);
}

void WebOpenPanelResultListenerProxy::cancel()
{
    if (!m_page)
        return;

    m_page->didCancelForOpenPanel();
}

void WebOpenPanelResultListenerProxy::invalidate()
{
    m_page = nullptr;
    m_process = nullptr;
}

WebProcessProxy* WebOpenPanelResultListenerProxy::process() const
{
    return m_process.get();
}

} // namespace WebKit
