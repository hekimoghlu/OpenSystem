/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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

#include "WebPopupItem.h"

#include "ArgumentCoders.h"

namespace WebKit {

WebPopupItem::WebPopupItem()
    : m_type(Type::Item)
    , m_textDirection(WebCore::TextDirection::LTR)
    , m_hasTextDirectionOverride(false)
    , m_isEnabled(true)
    , m_isSelected(false)
{
}

WebPopupItem::WebPopupItem(Type type)
    : m_type(type)
    , m_textDirection(WebCore::TextDirection::LTR)
    , m_hasTextDirectionOverride(false)
    , m_isEnabled(true)
    , m_isLabel(false)
    , m_isSelected(false)
{
}

WebPopupItem::WebPopupItem(Type type, const String& text, WebCore::TextDirection textDirection, bool hasTextDirectionOverride, const String& toolTip, const String& accessibilityText, bool isEnabled, bool isLabel, bool isSelected)
    : m_type(type)
    , m_text(text)
    , m_textDirection(textDirection)
    , m_hasTextDirectionOverride(hasTextDirectionOverride)
    , m_toolTip(toolTip)
    , m_accessibilityText(accessibilityText)
    , m_isEnabled(isEnabled)
    , m_isLabel(isLabel)
    , m_isSelected(isSelected)
{
}

} // namespace WebKit
