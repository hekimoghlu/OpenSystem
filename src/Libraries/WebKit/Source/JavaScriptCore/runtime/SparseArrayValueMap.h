/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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

#include "JSCast.h"
#include "JSTypeInfo.h"
#include "PropertyDescriptor.h"
#include "PutDirectIndexMode.h"
#include "VM.h"
#include "WriteBarrier.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class SparseArrayEntry;

class SparseArrayValueMap final : public JSCell {
public:
    typedef JSCell Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;
    
private:
    typedef UncheckedKeyHashMap<uint64_t, SparseArrayEntry, WTF::IntHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> Map;

    enum Flags {
        Normal                             = 0,
        SparseMode                         = 1 << 0,
        LengthIsReadOnly                   = 1 << 1,
        HasAnyKindOfGetterSetterProperties = 1 << 2,
    };

    SparseArrayValueMap(VM&);
    
    DECLARE_DEFAULT_FINISH_CREATION;

public:
    DECLARE_EXPORT_INFO;
    
    typedef Map::iterator iterator;
    typedef Map::const_iterator const_iterator;
    typedef Map::AddResult AddResult;

    static SparseArrayValueMap* create(VM&);
    
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.sparseArrayValueMapSpace();
    }
    
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);

    DECLARE_VISIT_CHILDREN;

    bool sparseMode()
    {
        return m_flags & SparseMode;
    }

    void setSparseMode()
    {
        m_flags = static_cast<Flags>(m_flags | SparseMode);
    }

    bool lengthIsReadOnly()
    {
        return m_flags & LengthIsReadOnly;
    }

    void setLengthIsReadOnly()
    {
        m_flags = static_cast<Flags>(m_flags | LengthIsReadOnly);
    }

    bool hasAnyKindOfGetterSetterProperties()
    {
        return m_flags & HasAnyKindOfGetterSetterProperties;
    }

    void setHasAnyKindOfGetterSetterProperties()
    {
        m_flags = static_cast<Flags>(m_flags | HasAnyKindOfGetterSetterProperties);
    }

    // These methods may mutate the contents of the map
    bool putEntry(JSGlobalObject*, JSObject*, unsigned, JSValue, bool shouldThrow);
    bool putDirect(JSGlobalObject*, JSObject*, unsigned, JSValue, unsigned attributes, PutDirectIndexMode);
    AddResult add(JSObject*, unsigned);
    iterator find(unsigned i) { return m_map.find(i); }
    // This should ASSERT the remove is valid (check the result of the find).
    void remove(iterator it);
    void remove(unsigned i);

    JSValue getConcurrently(unsigned index);

    // These methods do not mutate the contents of the map.
    iterator notFound() { return m_map.end(); }
    bool isEmpty() const { return m_map.isEmpty(); }
    bool contains(unsigned i) const { return m_map.contains(i); }
    size_t size() const { return m_map.size(); }
    // Only allow const begin/end iteration.
    const_iterator begin() const { return m_map.begin(); }
    const_iterator end() const { return m_map.end(); }

private:
    Map m_map;
    Flags m_flags { Normal };
    size_t m_reportedCapacity { 0 };
};

class SparseArrayEntry : private WriteBarrier<Unknown> {
    WTF_MAKE_TZONE_ALLOCATED(SparseArrayEntry);
public:
    using Base = WriteBarrier<Unknown>;

    SparseArrayEntry()
    {
        Base::setWithoutWriteBarrier(jsUndefined());
    }

    void get(JSObject*, PropertySlot&) const;
    void get(PropertyDescriptor&) const;
    bool put(JSGlobalObject*, JSValue thisValue, SparseArrayValueMap*, JSValue, bool shouldThrow);
    JSValue getNonSparseMode() const;
    JSValue getConcurrently() const;
    JSValue get() const;

    unsigned attributes() const { return m_attributes; }

    void forceSet(SparseArrayValueMap* map, unsigned attributes)
    {
        // FIXME: We can expand this for non x86 environments. Currently, loading ReadOnly | DontDelete property
        // from compiler thread is only supported in X86 architecture because of its TSO nature.
        // https://bugs.webkit.org/show_bug.cgi?id=134641
        if (isX86())
            WTF::storeStoreFence();

        if (attributes & PropertyAttribute::Accessor)
            map->setHasAnyKindOfGetterSetterProperties();
        m_attributes = attributes;
    }

    void forceSet(VM& vm, SparseArrayValueMap* map, JSValue value, unsigned attributes)
    {
        Base::set(vm, map, value);
        forceSet(map, attributes);
    }

    WriteBarrier<Unknown>& asValue() { return *this; }

private:
    unsigned m_attributes { 0 };
};

} // namespace JSC
