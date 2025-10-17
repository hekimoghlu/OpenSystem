/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

#include <wtf/Forward.h>
#include <wtf/Markable.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

using SharedStringHash = uint32_t;

// This is a hash value, but it can be used as a key in UncheckedKeyHashMap. So, we need to avoid producing deleted-value in UncheckedKeyHashMap, which is -1.
struct SharedStringHashHash {
    static unsigned hash(SharedStringHash key) { return static_cast<unsigned>(key); }
    static bool equal(SharedStringHash a, SharedStringHash b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
    static constexpr SharedStringHash deletedValue = std::numeric_limits<SharedStringHash>::max();
};

using SharedStringHashMarkableTraits = IntegralMarkableTraits<SharedStringHash, SharedStringHashHash::deletedValue>;

// Returns the hash of the string that will be used for visited link coloring.
WEBCORE_EXPORT SharedStringHash computeSharedStringHash(const String& url);
WEBCORE_EXPORT SharedStringHash computeSharedStringHash(std::span<const UChar> url);

// Resolves the potentially relative URL "attributeURL" relative to the given
// base URL, and returns the hash of the string that will be used for visited
// link coloring. It will return the special value of 0 if attributeURL does not
// look like a relative URL.
SharedStringHash computeVisitedLinkHash(const URL& base, const AtomString& attributeURL);

} // namespace WebCore
