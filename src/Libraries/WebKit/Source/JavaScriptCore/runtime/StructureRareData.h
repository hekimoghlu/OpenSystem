/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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

#include "ClassInfo.h"
#include "JSCast.h"
#include "JSTypeInfo.h"
#include "PropertyOffset.h"
#include "PropertySlot.h"
#include <wtf/FixedVector.h>

namespace JSC {

class JSPropertyNameEnumerator;
class LLIntOffsetsExtractor;
class Structure;
class StructureChain;
class CachedSpecialPropertyAdaptiveStructureWatchpoint;
class CachedSpecialPropertyAdaptiveInferredPropertyValueWatchpoint;
struct SpecialPropertyCache;
enum class CachedPropertyNamesKind : uint8_t {
    EnumerableStrings = 0,
    Strings,
    Symbols,
    StringsAndSymbols,
};
static constexpr unsigned numberOfCachedPropertyNames = 4;

enum class CachedSpecialPropertyKey : uint8_t {
    ToStringTag = 0,
    ToString,
    ValueOf,
    ToPrimitive,
    ToJSON,
};
static constexpr unsigned numberOfCachedSpecialPropertyKeys = 5;

class StructureRareData;
class StructureChainInvalidationWatchpoint;

class StructureRareData final : public JSCell {
public:
    typedef JSCell Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.structureRareDataSpace();
    }

    static StructureRareData* create(VM&, Structure*);

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    DECLARE_VISIT_CHILDREN;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);

    Structure* previousID() const
    {
        return m_previous.get();
    }
    void setPreviousID(VM&, Structure*);
    void clearPreviousID();

    JSValue cachedSpecialProperty(CachedSpecialPropertyKey) const;
    void cacheSpecialProperty(JSGlobalObject*, VM&, Structure* baseStructure, JSValue, CachedSpecialPropertyKey, const PropertySlot&);

    JSPropertyNameEnumerator* cachedPropertyNameEnumerator() const;
    uintptr_t cachedPropertyNameEnumeratorAndFlag() const;
    void setCachedPropertyNameEnumerator(VM&, Structure*, JSPropertyNameEnumerator*, StructureChain*);
    void clearCachedPropertyNameEnumerator();

    JSImmutableButterfly* cachedPropertyNames(CachedPropertyNamesKind) const;
    JSImmutableButterfly* cachedPropertyNamesIgnoringSentinel(CachedPropertyNamesKind) const;
    JSImmutableButterfly* cachedPropertyNamesConcurrently(CachedPropertyNamesKind) const;
    void setCachedPropertyNames(VM&, CachedPropertyNamesKind, JSImmutableButterfly*);

    Box<InlineWatchpointSet> copySharedPolyProtoWatchpoint() const { return m_polyProtoWatchpoint; }
    const Box<InlineWatchpointSet>& sharedPolyProtoWatchpoint() const { return m_polyProtoWatchpoint; }
    void setSharedPolyProtoWatchpoint(Box<InlineWatchpointSet>&& sharedPolyProtoWatchpoint) { m_polyProtoWatchpoint = WTFMove(sharedPolyProtoWatchpoint); }
    bool hasSharedPolyProtoWatchpoint() const { return static_cast<bool>(m_polyProtoWatchpoint); }

    static JSImmutableButterfly* cachedPropertyNamesSentinel() { return std::bit_cast<JSImmutableButterfly*>(static_cast<uintptr_t>(1)); }

    static constexpr ptrdiff_t offsetOfCachedPropertyNames(CachedPropertyNamesKind kind)
    {
        return OBJECT_OFFSETOF(StructureRareData, m_cachedPropertyNames) + sizeof(WriteBarrier<JSImmutableButterfly>) * static_cast<unsigned>(kind);
    }

    static constexpr ptrdiff_t offsetOfCachedPropertyNameEnumeratorAndFlag()
    {
        return OBJECT_OFFSETOF(StructureRareData, m_cachedPropertyNameEnumeratorAndFlag);
    }

    static constexpr ptrdiff_t offsetOfSpecialPropertyCache()
    {
        return OBJECT_OFFSETOF(StructureRareData, m_specialPropertyCache);
    }

    static constexpr ptrdiff_t offsetOfPrevious()
    {
        return OBJECT_OFFSETOF(StructureRareData, m_previous);
    }

    DECLARE_EXPORT_INFO;

    void finalizeUnconditionally(VM&, CollectionScope);

    static constexpr uintptr_t cachedPropertyNameEnumeratorIsValidatedViaTraversingFlag = 1;
    static constexpr uintptr_t cachedPropertyNameEnumeratorMask = ~static_cast<uintptr_t>(1);

    unsigned incrementActiveReplacementWatchpointSet()
    {
        return ++m_activeReplacementWatchpointSet;
    }

    unsigned decrementActiveReplacementWatchpointSet()
    {
        return --m_activeReplacementWatchpointSet;
    }

private:
    friend class LLIntOffsetsExtractor;
    friend class Structure;
    friend class CachedSpecialPropertyAdaptiveStructureWatchpoint;
    friend class CachedSpecialPropertyAdaptiveInferredPropertyValueWatchpoint;

    StructureRareData(VM&, Structure*);

    void clearCachedSpecialProperty(CachedSpecialPropertyKey);
    void cacheSpecialPropertySlow(JSGlobalObject*, VM&, Structure* baseStructure, JSValue, CachedSpecialPropertyKey, const PropertySlot&);

    SpecialPropertyCache& ensureSpecialPropertyCache();
    SpecialPropertyCache& ensureSpecialPropertyCacheSlow();
    bool canCacheSpecialProperty(CachedSpecialPropertyKey);
    void giveUpOnSpecialPropertyCache(CachedSpecialPropertyKey);

    bool tryCachePropertyNameEnumeratorViaWatchpoint(VM&, Structure*, StructureChain*);

    // FIXME: We should have some story for clearing these property names caches in GC.
    // https://bugs.webkit.org/show_bug.cgi?id=192659
    uintptr_t m_cachedPropertyNameEnumeratorAndFlag { 0 };
    FixedVector<StructureChainInvalidationWatchpoint> m_cachedPropertyNameEnumeratorWatchpoints;
    WriteBarrier<JSImmutableButterfly> m_cachedPropertyNames[numberOfCachedPropertyNames] { };

    typedef UncheckedKeyHashMap<PropertyOffset, RefPtr<WatchpointSet>, WTF::IntHash<PropertyOffset>, WTF::UnsignedWithZeroKeyHashTraits<PropertyOffset>> PropertyWatchpointMap;
#ifdef NDEBUG
    static_assert(sizeof(PropertyWatchpointMap) == sizeof(void*), "StructureRareData should remain small");
#endif

    PropertyWatchpointMap m_replacementWatchpointSets;
    std::unique_ptr<SpecialPropertyCache> m_specialPropertyCache;
    Box<InlineWatchpointSet> m_polyProtoWatchpoint;

    WriteBarrierStructureID m_previous;
    PropertyOffset m_maxOffset;
    PropertyOffset m_transitionOffset;
    unsigned m_activeReplacementWatchpointSet { 0 };
};

} // namespace JSC
