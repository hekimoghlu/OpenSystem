/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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

#include "Color.h"
#include <wtf/HashTraits.h>

namespace WTF {

struct ColorHash {
    static unsigned hash(const WebCore::Color& key) { return computeHash(key); }
    static bool equal(const WebCore::Color& a, const WebCore::Color& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct DefaultHash<WebCore::Color> : ColorHash { };

template<> struct HashTraits<WebCore::Color> : GenericHashTraits<WebCore::Color> {
    static const bool emptyValueIsZero = false;
    static WebCore::Color emptyValue() { return WebCore::Color(HashTableEmptyValue); }
    static bool isEmptyValue(const WebCore::Color& value) { return value.isHashTableEmptyValue(); }

    static void constructDeletedValue(WebCore::Color& slot) { new (NotNull, &slot) WebCore::Color(HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::Color& value) { return value.isHashTableDeletedValue(); }
};

} // namespace WTF
