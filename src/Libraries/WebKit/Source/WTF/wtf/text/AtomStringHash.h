/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include <wtf/text/AtomString.h>
#include <wtf/HashTraits.h>

namespace WTF {

    struct AtomStringHash {
        static unsigned hash(const AtomString& key)
        {
            return key.impl()->existingHash();
        }

        static bool equal(const AtomString& a, const AtomString& b)
        {
            return a == b;
        }

        static constexpr bool safeToCompareToEmptyOrDeleted = false;
        static constexpr bool hasHashInValue = true;
    };

    template<> struct HashTraits<WTF::AtomString> : SimpleClassHashTraits<WTF::AtomString> {
        static constexpr bool hasIsEmptyValueFunction = true;
        static bool isEmptyValue(const AtomString& value)
        {
            return value.isNull();
        }

        static void customDeleteBucket(AtomString& value)
        {
            // See unique_ptr's customDeleteBucket() for an explanation.
            ASSERT(!isDeletedValue(value));
            AtomString valueToBeDestroyed = WTFMove(value);
            constructDeletedValue(value);
        }
    };

    template<> struct DefaultHash<AtomString> : AtomStringHash { };
}

using WTF::AtomStringHash;
