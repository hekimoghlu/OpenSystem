/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#pragma once

#include "WebKitOptionMenuItem.h"
#include "WebPopupItem.h"
#include <wtf/text/CString.h>

struct _WebKitOptionMenuItem {
    _WebKitOptionMenuItem() = default;

    _WebKitOptionMenuItem(const WebKit::WebPopupItem& item)
        : label(item.m_text.trim(deprecatedIsSpaceOrNewline).utf8())
        , isGroupLabel(item.m_isLabel)
        , isGroupChild(item.m_text.startsWith("    "_s))
        , isEnabled(item.m_isEnabled)
    {
        if (!item.m_toolTip.isEmpty())
            tooltip = item.m_toolTip.utf8();
    }

    explicit _WebKitOptionMenuItem(_WebKitOptionMenuItem* other)
        : label(other->label)
        , tooltip(other->tooltip)
        , isGroupLabel(other->isGroupLabel)
        , isGroupChild(other->isGroupChild)
        , isEnabled(other->isEnabled)
        , isSelected(other->isSelected)
    {
    }

    CString label;
    CString tooltip;
    bool isGroupLabel { false };
    bool isGroupChild { false };
    bool isEnabled { true };
    bool isSelected { false };
};
