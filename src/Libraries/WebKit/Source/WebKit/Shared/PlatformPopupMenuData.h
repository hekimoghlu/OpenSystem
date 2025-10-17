/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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

#include "PlatformFontInfo.h"
#include <WebCore/PopupMenuStyle.h>
#include <WebCore/ShareableBitmap.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

struct PlatformPopupMenuData {
#if PLATFORM(COCOA)
    WebKit::PlatformFontInfo fontInfo;
    bool shouldPopOver { false };
    bool hideArrows { false };
    WebCore::PopupMenuStyle::Size menuSize { WebCore::PopupMenuStyle::Size::Normal };
#elif PLATFORM(WIN)
    int m_clientPaddingLeft { 0 };
    int m_clientPaddingRight { 0 };
    int m_clientInsetLeft { 0 };
    int m_clientInsetRight { 0 };
    int m_popupWidth { 0 };
    float m_itemHeight { 0 };
    bool m_isRTL { false };
    RefPtr<WebCore::ShareableBitmap> m_notSelectedBackingStore;
    RefPtr<WebCore::ShareableBitmap> m_selectedBackingStore;
#endif
};

} // namespace WebKit
