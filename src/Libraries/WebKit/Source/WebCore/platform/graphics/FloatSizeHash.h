/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#ifndef FloatSizeHash_h
#define FloatSizeHash_h

#include "FloatSize.h"
#include <wtf/HashSet.h>

namespace WTF {

template<> struct FloatHash<WebCore::FloatSize> {
    static unsigned hash(const WebCore::FloatSize& key) { return pairIntHash(DefaultHash<float>::hash(key.width()), DefaultHash<float>::hash(key.height())); }
    static bool equal(const WebCore::FloatSize& a, const WebCore::FloatSize& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct DefaultHash<WebCore::FloatSize> : FloatHash<WebCore::FloatSize> { };

template<> struct HashTraits<WebCore::FloatSize> : GenericHashTraits<WebCore::FloatSize> {
    static const bool emptyValueIsZero = true;
    static void constructDeletedValue(WebCore::FloatSize& slot) { new (NotNull, &slot) WebCore::FloatSize(-1, -1); }
    static bool isDeletedValue(const WebCore::FloatSize& value) { return value.width() == -1 && value.height() == -1; }
};

} // namespace WTF

#endif
