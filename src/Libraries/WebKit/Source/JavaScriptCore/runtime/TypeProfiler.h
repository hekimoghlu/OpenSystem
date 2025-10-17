/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#include "TypeLocationCache.h"
#include <wtf/Bag.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace Inspector { namespace Protocol  { namespace Runtime {
class TypeDescription;
}}}

namespace JSC {

enum TypeProfilerSearchDescriptor {
    TypeProfilerSearchDescriptorNormal = 1,
    TypeProfilerSearchDescriptorFunctionReturn = 2
};

struct QueryKey {
    QueryKey()
        : m_sourceID(0)
        , m_divot(0)
        , m_searchDescriptor(TypeProfilerSearchDescriptorFunctionReturn)
    { }

    QueryKey(SourceID sourceID, unsigned divot, TypeProfilerSearchDescriptor searchDescriptor)
        : m_sourceID(sourceID)
        , m_divot(divot)
        , m_searchDescriptor(searchDescriptor)
    { }

    QueryKey(WTF::HashTableDeletedValueType)
        : m_sourceID(UINT_MAX)
        , m_divot(UINT_MAX)
        , m_searchDescriptor(TypeProfilerSearchDescriptorFunctionReturn)
    { }

    bool isHashTableDeletedValue() const 
    { 
        return m_sourceID == UINT_MAX
            && m_divot == UINT_MAX
            && m_searchDescriptor == TypeProfilerSearchDescriptorFunctionReturn;
    }

    friend bool operator==(const QueryKey&, const QueryKey&) = default;

    unsigned hash() const 
    { 
        unsigned hash = static_cast<unsigned>(m_sourceID) + m_divot * m_searchDescriptor;
        return hash;
    }

    SourceID m_sourceID;
    unsigned m_divot;
    TypeProfilerSearchDescriptor m_searchDescriptor;
};

struct QueryKeyHash {
    static unsigned hash(const QueryKey& key) { return key.hash(); }
    static bool equal(const QueryKey& a, const QueryKey& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} // namespace JSC

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::QueryKey> : JSC::QueryKeyHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::QueryKey> : SimpleClassHashTraits<JSC::QueryKey> {
    static constexpr bool emptyValueIsZero = false;
};

} // namespace WTF

namespace JSC {

class VM;

class TypeProfiler {
    WTF_MAKE_TZONE_ALLOCATED(TypeProfiler);
public:
    TypeProfiler();
    void logTypesForTypeLocation(TypeLocation*, VM&);
    JS_EXPORT_PRIVATE String typeInformationForExpressionAtOffset(TypeProfilerSearchDescriptor, unsigned offset, SourceID, VM&);
    void insertNewLocation(TypeLocation*);
    TypeLocationCache* typeLocationCache() { return &m_typeLocationCache; }
    TypeLocation* findLocation(unsigned divot, SourceID, TypeProfilerSearchDescriptor, VM&);
    GlobalVariableID getNextUniqueVariableID() { return m_nextUniqueVariableID++; }
    TypeLocation* nextTypeLocation();
    void invalidateTypeSetCache(VM&);
    void dumpTypeProfilerData(VM&);
    
private:
    typedef UncheckedKeyHashMap<SourceID, Vector<TypeLocation*>> SourceIDToLocationBucketMap;
    SourceIDToLocationBucketMap m_bucketMap;
    TypeLocationCache m_typeLocationCache;
    typedef UncheckedKeyHashMap<QueryKey, TypeLocation*> TypeLocationQueryCache;
    TypeLocationQueryCache m_queryCache;
    GlobalVariableID m_nextUniqueVariableID;
    Bag<TypeLocation> m_typeLocationInfo;
};

} // namespace JSC
