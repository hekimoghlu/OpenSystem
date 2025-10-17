/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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

#include <wtf/Hasher.h>
#include <wtf/Vector.h>

namespace WebCore {

namespace ContentExtensions {

struct HashableActionList {
    enum DeletedValueTag { DeletedValue };
    explicit HashableActionList(DeletedValueTag) { state = Deleted; }

    enum EmptyValueTag { EmptyValue };
    explicit HashableActionList(EmptyValueTag) { state = Empty; }

    template<typename AnyVectorType>
    explicit HashableActionList(const AnyVectorType& otherActions)
        : actions(otherActions)
        , state(Valid)
    {
        std::sort(actions.begin(), actions.end());
        SuperFastHash hasher;
        hasher.addCharactersAssumingAligned(reinterpret_cast<const UChar*>(actions.data()), actions.size() * sizeof(uint64_t) / sizeof(UChar));
        hash = hasher.hash();
    }

    bool isEmptyValue() const { return state == Empty; }
    bool isDeletedValue() const { return state == Deleted; }

    bool operator==(const HashableActionList& other) const
    {
        return state == other.state && actions == other.actions;
    }

    Vector<uint64_t> actions;
    unsigned hash;
    enum {
        Valid,
        Empty,
        Deleted
    } state;
};

struct HashableActionListHash {
    static unsigned hash(const HashableActionList& actionKey)
    {
        return actionKey.hash;
    }

    static bool equal(const HashableActionList& a, const HashableActionList& b)
    {
        return a == b;
    }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

struct HashableActionListHashTraits : public WTF::CustomHashTraits<HashableActionList> {
    static const bool emptyValueIsZero = false;
};

} // namespace ContentExtensions
} // namespace WebCore
