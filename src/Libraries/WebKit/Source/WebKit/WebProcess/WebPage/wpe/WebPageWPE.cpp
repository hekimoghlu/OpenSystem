/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

#include "DrawingArea.h"
#include "WebPageProxy.h"
#include "WebPageProxyMessages.h"
#include <WebCore/NotImplemented.h>
#include <WebCore/PlatformScreen.h>
#include <WebCore/PointerCharacteristics.h>

namespace WebKit {
using namespace WebCore;

void WebPage::platformReinitialize()
{
}

bool WebPage::platformCanHandleRequest(const ResourceRequest&)
{
    notImplemented();
    return false;
}

bool WebPage::hoverSupportedByPrimaryPointingDevice() const
{
#if ENABLE(TOUCH_EVENTS)
    return !screenIsTouchPrimaryInputDevice();
#else
    return true;
#endif
}

bool WebPage::hoverSupportedByAnyAvailablePointingDevice() const
{
#if ENABLE(TOUCH_EVENTS)
    return !screenHasTouchDevice();
#else
    return true;
#endif
}

std::optional<PointerCharacteristics> WebPage::pointerCharacteristicsOfPrimaryPointingDevice() const
{
#if ENABLE(TOUCH_EVENTS)
    if (screenIsTouchPrimaryInputDevice())
        return PointerCharacteristics::Coarse;
#endif
    return PointerCharacteristics::Fine;
}

OptionSet<PointerCharacteristics> WebPage::pointerCharacteristicsOfAllAvailablePointingDevices() const
{
#if ENABLE(TOUCH_EVENTS)
    if (screenHasTouchDevice())
        return PointerCharacteristics::Coarse;
#endif
    return PointerCharacteristics::Fine;
}

#if USE(GBM) && ENABLE(WPE_PLATFORM)
void WebPage::preferredBufferFormatsDidChange(Vector<DMABufRendererBufferFormat>&& preferredBufferFormats)
{
    m_preferredBufferFormats = WTFMove(preferredBufferFormats);
    if (m_drawingArea)
        m_drawingArea->preferredBufferFormatsDidChange();
}
#endif

} // namespace WebKit
