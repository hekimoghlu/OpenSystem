/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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

#include "IntPointHash.h"
#include "IntRect.h"
#include "IntSizeHash.h"

namespace WTF {

template<> struct IntHash<WebCore::IntRect> {
    static unsigned hash(const WebCore::IntRect& key)
    {
        return pairIntHash(DefaultHash<WebCore::IntPoint>::hash(key.location()), DefaultHash<WebCore::IntSize>::hash(key.size()));
    }
    static bool equal(const WebCore::IntRect& a, const WebCore::IntRect& b)
    {
        return DefaultHash<WebCore::IntPoint>::equal(a.location(), b.location()) && DefaultHash<WebCore::IntSize>::equal(a.size(), b.size());
    }
    static const bool safeToCompareToEmptyOrDeleted = true;
};
template<> struct DefaultHash<WebCore::IntRect> : IntHash<WebCore::IntRect> { };

template<> struct HashTraits<WebCore::IntRect> : GenericHashTraits<WebCore::IntRect> {
    static const bool emptyValueIsZero = true;
    static void constructDeletedValue(WebCore::IntRect& slot) { new (NotNull, &slot) WebCore::IntRect(-1, -1, -1, -1); }
    static bool isDeletedValue(const WebCore::IntRect& value) { return value.x() == -1 && value.y() == -1 && value.width() == -1 && value.height() == -1; }
};

}
