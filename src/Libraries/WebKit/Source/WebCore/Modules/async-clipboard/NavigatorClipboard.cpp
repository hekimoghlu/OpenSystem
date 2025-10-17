/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
#include "NavigatorClipboard.h"

#include "Clipboard.h"
#include "Navigator.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorClipboard);

NavigatorClipboard::NavigatorClipboard(Navigator& navigator)
    : m_navigator(navigator)
{
}

NavigatorClipboard::~NavigatorClipboard() = default;

RefPtr<Clipboard> NavigatorClipboard::clipboard(Navigator& navigator)
{
    return NavigatorClipboard::from(navigator)->clipboard();
}

RefPtr<Clipboard> NavigatorClipboard::clipboard()
{
    if (!m_clipboard)
        m_clipboard = Clipboard::create(Ref { m_navigator.get() });
    return m_clipboard;
}

NavigatorClipboard* NavigatorClipboard::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorClipboard*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorClipboard>(navigator);
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral NavigatorClipboard::supplementName()
{
    return "NavigatorClipboard"_s;
}

}
