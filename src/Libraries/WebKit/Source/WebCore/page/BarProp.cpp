/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include "BarProp.h"

#include "Chrome.h"
#include "LocalFrame.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(BarProp);

BarProp::BarProp(LocalDOMWindow& window, Type type)
    : LocalDOMWindowProperty(&window)
    , m_type(type)
{
}

bool BarProp::visible() const
{
    auto* frame = this->frame();
    if (!frame)
        return false;
    RefPtr page = frame->page();
    if (!page)
        return false;

    switch (m_type) {
    case Locationbar:
    case Personalbar:
    case Toolbar:
        return page->chrome().toolbarsVisible();
    case Menubar:
        return page->chrome().menubarVisible();
    case Scrollbars:
        return page->chrome().scrollbarsVisible();
    case Statusbar:
        return page->chrome().statusbarVisible();
    }

    ASSERT_NOT_REACHED();
    return false;
}

} // namespace WebCore
