/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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
#ifndef IntSizeHash_h
#define IntSizeHash_h

#include "IntSize.h"
#include <wtf/HashTraits.h>

namespace WTF {

    template<> struct IntHash<WebCore::IntSize> {
        static unsigned hash(const WebCore::IntSize& key) { return pairIntHash(key.width(), key.height()); }
        static bool equal(const WebCore::IntSize& a, const WebCore::IntSize& b) { return a == b; }
        static const bool safeToCompareToEmptyOrDeleted = true;
    };
    template<> struct DefaultHash<WebCore::IntSize> : IntHash<WebCore::IntSize> { };
    
    template<> struct HashTraits<WebCore::IntSize> : GenericHashTraits<WebCore::IntSize> {
        static const bool emptyValueIsZero = true;
        static void constructDeletedValue(WebCore::IntSize& slot) { new (NotNull, &slot) WebCore::IntSize(-1, -1); }
        static bool isDeletedValue(const WebCore::IntSize& value) { return value.width() == -1 && value.height() == -1; }
    };
} // namespace WTF

#endif
