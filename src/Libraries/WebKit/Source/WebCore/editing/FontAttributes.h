/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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

#include "Font.h"
#include "FontShadow.h"
#include "ListStyleType.h"
#include "RenderStyleConstants.h"
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSDictionary;
OBJC_CLASS NSTextList;

namespace WebCore {

struct TextList {
    ListStyleType styleType { ListStyleType::Type::None, nullAtom() };
    int startingItemNumber { 0 };
    bool ordered { false };

#if PLATFORM(COCOA)
    RetainPtr<NSTextList> createTextList() const;
#endif
};

struct FontAttributes {
    enum class SubscriptOrSuperscript : uint8_t { None, Subscript, Superscript };
    enum class HorizontalAlignment : uint8_t { Left, Center, Right, Justify, Natural };

#if PLATFORM(COCOA)
    WEBCORE_EXPORT RetainPtr<NSDictionary> createDictionary() const;
#endif

    RefPtr<Font> font;
    Color backgroundColor;
    Color foregroundColor;
    FontShadow fontShadow;
    SubscriptOrSuperscript subscriptOrSuperscript { SubscriptOrSuperscript::None };
    HorizontalAlignment horizontalAlignment { HorizontalAlignment::Left };
    Vector<TextList> textLists;
    bool hasUnderline { false };
    bool hasStrikeThrough { false };
    bool hasMultipleFonts { false };
};

} // namespace WebCore
