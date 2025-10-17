/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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
#ifndef IntPointHash_h
#define IntPointHash_h

#include "IntPoint.h"
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>

namespace WTF {
    
// The empty value is (0, INT_MIN), the deleted value is (INT_MIN, 0)
struct IntPointHash {
    static unsigned hash(const WebCore::IntPoint& p) { return pairIntHash(p.x(), p.y()); }
    static bool equal(const WebCore::IntPoint& a, const WebCore::IntPoint& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};
template<> struct HashTraits<WebCore::IntPoint> : GenericHashTraits<WebCore::IntPoint> {
    static WebCore::IntPoint emptyValue() { return WebCore::IntPoint(0, std::numeric_limits<int>::min()); }
    static bool isEmptyValue(const WebCore::IntPoint& value) { return value.y() == std::numeric_limits<int>::min(); }
    
    static void constructDeletedValue(WebCore::IntPoint& slot) { slot.setX(std::numeric_limits<int>::min()); }
    static bool isDeletedValue(const WebCore::IntPoint& slot) { return slot.x() == std::numeric_limits<int>::min(); }
};
template<> struct DefaultHash<WebCore::IntPoint> : IntPointHash { };

}

#endif
