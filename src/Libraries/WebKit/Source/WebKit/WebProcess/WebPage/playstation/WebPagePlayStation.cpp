/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
#include "WebPage.h"

#include <WebCore/NotImplemented.h>
#include <WebCore/Page.h>
#include <WebCore/PointerCharacteristics.h>
#include <WebCore/Settings.h>
#include <WebCore/UserAgent.h>

namespace WebKit {
using namespace WebCore;

void WebPage::platformInitialize(const WebPageCreationParameters&)
{
}

void WebPage::platformReinitialize()
{
}

void WebPage::platformDetach()
{
}

bool WebPage::platformCanHandleRequest(const ResourceRequest&)
{
    notImplemented();
    return false;
}

String WebPage::platformUserAgent(const URL& url) const
{
    if (url.isNull() || !m_page->settings().needsSiteSpecificQuirks())
        return emptyString();

    return WebCore::standardUserAgentForURL(url);
}

bool WebPage::hoverSupportedByPrimaryPointingDevice() const
{
    return true;
}

bool WebPage::hoverSupportedByAnyAvailablePointingDevice() const
{
    return true;
}

std::optional<PointerCharacteristics> WebPage::pointerCharacteristicsOfPrimaryPointingDevice() const
{
    return PointerCharacteristics::Fine;
}

OptionSet<PointerCharacteristics> WebPage::pointerCharacteristicsOfAllAvailablePointingDevices() const
{
    return PointerCharacteristics::Fine;
}

bool WebPage::handleEditingKeyboardEvent(WebCore::KeyboardEvent& event)
{
    notImplemented();
    return false;
}

void WebPage::getPlatformEditorState(LocalFrame& frame, EditorState& result) const
{
    notImplemented();
}

} // namespace WebKit
