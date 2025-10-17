/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
#include "NavigatorMediaDevices.h"

#if ENABLE(MEDIA_STREAM)

#include "Document.h"
#include "LocalFrame.h"
#include "MediaDevices.h"
#include "Navigator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorMediaDevices);

NavigatorMediaDevices::NavigatorMediaDevices(LocalDOMWindow* window)
    : LocalDOMWindowProperty(window)
{
}

NavigatorMediaDevices::~NavigatorMediaDevices() = default;

NavigatorMediaDevices* NavigatorMediaDevices::from(Navigator* navigator)
{
    NavigatorMediaDevices* supplement = static_cast<NavigatorMediaDevices*>(Supplement<Navigator>::from(navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorMediaDevices>(navigator->window());
        supplement = newSupplement.get();
        provideTo(navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

MediaDevices* NavigatorMediaDevices::mediaDevices(Navigator& navigator)
{
    return NavigatorMediaDevices::from(&navigator)->mediaDevices();
}

MediaDevices* NavigatorMediaDevices::mediaDevices() const
{
    if (!m_mediaDevices && frame())
        m_mediaDevices = MediaDevices::create(*frame()->document());
    return m_mediaDevices.get();
}

ASCIILiteral NavigatorMediaDevices::supplementName()
{
    return "NavigatorMediaDevices"_s;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
