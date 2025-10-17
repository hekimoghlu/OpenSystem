/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#include "NavigatorMediaCapabilities.h"

#include "MediaCapabilities.h"
#include "Navigator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorMediaCapabilities);

NavigatorMediaCapabilities::NavigatorMediaCapabilities()
    : m_mediaCapabilities(MediaCapabilities::create())
{
}

NavigatorMediaCapabilities::~NavigatorMediaCapabilities() = default;

ASCIILiteral NavigatorMediaCapabilities::supplementName()
{
    return "NavigatorMediaCapabilities"_s;
}

NavigatorMediaCapabilities& NavigatorMediaCapabilities::from(Navigator& navigator)
{
    NavigatorMediaCapabilities* supplement = static_cast<NavigatorMediaCapabilities*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorMediaCapabilities>();
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return *supplement;
}

MediaCapabilities& NavigatorMediaCapabilities::mediaCapabilities(Navigator& navigator)
{
    return NavigatorMediaCapabilities::from(navigator).mediaCapabilities();
}

MediaCapabilities& NavigatorMediaCapabilities::mediaCapabilities() const
{
    return m_mediaCapabilities;
}

} // namespace WebCore
