/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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

#include "SourceID.h"
#include "TypeLocation.h"
#include <wtf/FastMalloc.h>
#include <wtf/GenericHashKey.h>

namespace JSC {

class VM;

class TypeLocationCache {
public:
    struct LocationKey {
        struct Hash {
            static unsigned hash(const LocationKey& key) { return key.hash(); }
            static bool equal(const LocationKey& a, const LocationKey& b) { return a == b; }
            static constexpr bool safeToCompareToEmptyOrDeleted = false;
        };

        LocationKey() {}
        friend bool operator==(const LocationKey&, const LocationKey&) = default;

        unsigned hash() const
        {
            return m_globalVariableID + m_sourceID + m_start + m_end;
        }

        GlobalVariableID m_globalVariableID;
        SourceID m_sourceID;
        unsigned m_start;
        unsigned m_end;
    };

    std::pair<TypeLocation*, bool> getTypeLocation(GlobalVariableID, SourceID, unsigned start, unsigned end, RefPtr<TypeSet>&&, VM*);
private:
    using LocationMap = UncheckedKeyHashMap<GenericHashKey<LocationKey, LocationKey::Hash>, TypeLocation*>;
    LocationMap m_locationMap;
};

} // namespace JSC
