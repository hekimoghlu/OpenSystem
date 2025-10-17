/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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

#include "Identifier.h"
#include <wtf/HashSet.h>
#include <wtf/Vector.h>

namespace JSC {

// FIXME: Rename to PropertyNameArray.
class PropertyNameArrayData : public RefCounted<PropertyNameArrayData> {
public:
    typedef Vector<Identifier, 20> PropertyNameVector;

    static Ref<PropertyNameArrayData> create() { return adoptRef(*new PropertyNameArrayData); }

    PropertyNameVector& propertyNameVector() { return m_propertyNameVector; }

private:
    PropertyNameArrayData()
    {
    }

    PropertyNameVector m_propertyNameVector;
};

// FIXME: Rename to PropertyNameArrayBuilder.
class PropertyNameArray {
public:
    PropertyNameArray(VM& vm, PropertyNameMode propertyNameMode, PrivateSymbolMode privateSymbolMode)
        : m_data(PropertyNameArrayData::create())
        , m_vm(vm)
        , m_propertyNameMode(propertyNameMode)
        , m_privateSymbolMode(privateSymbolMode)
    {
    }

    VM& vm() { return m_vm; }

    void add(uint32_t index)
    {
        add(Identifier::from(m_vm, index));
    }

    void add(const Identifier&);
    void add(UniquedStringImpl*);
    void addUnchecked(UniquedStringImpl*);

    Identifier& operator[](unsigned i) { return m_data->propertyNameVector()[i]; }
    const Identifier& operator[](unsigned i) const { return m_data->propertyNameVector()[i]; }

    PropertyNameArrayData* data() { return m_data.get(); }
    RefPtr<PropertyNameArrayData> releaseData() { return WTFMove(m_data); }

    // FIXME: Remove these functions.
    bool canAddKnownUniqueForStructure() const { return m_data->propertyNameVector().isEmpty(); }
    typedef PropertyNameArrayData::PropertyNameVector::const_iterator const_iterator;
    size_t size() const { return m_data->propertyNameVector().size(); }
    const_iterator begin() const { return m_data->propertyNameVector().begin(); }
    const_iterator end() const { return m_data->propertyNameVector().end(); }

    bool includeSymbolProperties() const;
    bool includeStringProperties() const;

    PropertyNameMode propertyNameMode() const { return m_propertyNameMode; }
    PrivateSymbolMode privateSymbolMode() const { return m_privateSymbolMode; }

private:
    void addUncheckedInternal(UniquedStringImpl*);
    bool isUidMatchedToTypeMode(UniquedStringImpl* identifier);

    RefPtr<PropertyNameArrayData> m_data;
    UncheckedKeyHashSet<UniquedStringImpl*> m_set;
    VM& m_vm;
    PropertyNameMode m_propertyNameMode;
    PrivateSymbolMode m_privateSymbolMode;
};

ALWAYS_INLINE void PropertyNameArray::add(const Identifier& identifier)
{
    add(identifier.impl());
}

ALWAYS_INLINE void PropertyNameArray::addUncheckedInternal(UniquedStringImpl* identifier)
{
    m_data->propertyNameVector().append(Identifier::fromUid(m_vm, identifier));
}

ALWAYS_INLINE void PropertyNameArray::addUnchecked(UniquedStringImpl* identifier)
{
    if (!isUidMatchedToTypeMode(identifier))
        return;
    addUncheckedInternal(identifier);
}

ALWAYS_INLINE void PropertyNameArray::add(UniquedStringImpl* identifier)
{
    static constexpr unsigned setThreshold = 20;

    ASSERT(identifier);

    if (!isUidMatchedToTypeMode(identifier))
        return;

    if (size() < setThreshold) {
        if (m_data->propertyNameVector().contains(identifier))
            return;
    } else {
        if (m_set.isEmpty()) {
            for (Identifier& name : m_data->propertyNameVector())
                m_set.add(name.impl());
        }
        if (!m_set.add(identifier).isNewEntry)
            return;
    }

    addUncheckedInternal(identifier);
}

ALWAYS_INLINE bool PropertyNameArray::isUidMatchedToTypeMode(UniquedStringImpl* identifier)
{
    if (identifier->isSymbol()) {
        if (!includeSymbolProperties())
            return false;
        if (UNLIKELY(m_privateSymbolMode == PrivateSymbolMode::Include))
            return true;
        return !static_cast<SymbolImpl*>(identifier)->isPrivate();
    }
    return includeStringProperties();
}

ALWAYS_INLINE bool PropertyNameArray::includeSymbolProperties() const
{
    return static_cast<std::underlying_type<PropertyNameMode>::type>(m_propertyNameMode) & static_cast<std::underlying_type<PropertyNameMode>::type>(PropertyNameMode::Symbols);
}

ALWAYS_INLINE bool PropertyNameArray::includeStringProperties() const
{
    return static_cast<std::underlying_type<PropertyNameMode>::type>(m_propertyNameMode) & static_cast<std::underlying_type<PropertyNameMode>::type>(PropertyNameMode::Strings);
}

} // namespace JSC
