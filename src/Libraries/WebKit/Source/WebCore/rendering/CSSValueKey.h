/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

#include "CSSValueKeywords.h"

namespace WebCore {

struct CSSValueKey {
    unsigned hash() const;

    friend bool operator==(const CSSValueKey&, const CSSValueKey&) = default;

    unsigned cssValueID;
    bool useDarkAppearance;
    bool useElevatedUserInterfaceLevel;
};

inline unsigned CSSValueKey::hash() const
{
    return cssValueID;
}

} // namespace WebCore

namespace WTF {

struct CSSValueKeyHash {
    static unsigned hash(const WebCore::CSSValueKey& key) { return key.hash(); }
    static bool equal(const WebCore::CSSValueKey& a, const WebCore::CSSValueKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebCore::CSSValueKey> : GenericHashTraits<WebCore::CSSValueKey> {
    static WebCore::CSSValueKey emptyValue() { return WebCore::CSSValueKey { WebCore::CSSValueInvalid, false, false}; }
    static void constructDeletedValue(WebCore::CSSValueKey& slot) { new (NotNull, &slot) WebCore::CSSValueKey { WebCore::CSSValueInvalid, true, true}; }
    static bool isDeletedValue(const WebCore::CSSValueKey& slot) { return slot.cssValueID == WebCore::CSSValueInvalid && slot.useDarkAppearance && slot.useElevatedUserInterfaceLevel; }
};

template<> struct DefaultHash<WebCore::CSSValueKey> : CSSValueKeyHash { };

} // namespace WTF
