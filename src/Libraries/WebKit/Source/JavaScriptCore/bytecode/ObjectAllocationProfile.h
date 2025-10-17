/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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

#include "VM.h"
#include "ObjectPrototype.h"
#include "SlotVisitor.h"
#include "WriteBarrier.h"

namespace JSC {

class FunctionRareData;

template<typename Derived>
class ObjectAllocationProfileBase {
    friend class LLIntOffsetsExtractor;
public:
    static constexpr ptrdiff_t offsetOfAllocator() { return OBJECT_OFFSETOF(ObjectAllocationProfileBase, m_allocator); }
    static constexpr ptrdiff_t offsetOfStructure() { return OBJECT_OFFSETOF(ObjectAllocationProfileBase, m_structure); }

    ObjectAllocationProfileBase() = default;

    bool isNull() { return !m_structure; }

    void initializeProfile(VM&, JSGlobalObject*, JSCell* owner, JSObject* prototype, unsigned inferredInlineCapacity, JSFunction* constructor = nullptr, FunctionRareData* = nullptr);

    Structure* structure()
    {
        Structure* structure = m_structure.get();
        // Ensure that if we see the structure, it has been properly created
        WTF::dependentLoadLoadFence();
        return structure;
    }

protected:
    void clear()
    {
        m_allocator = Allocator();
        m_structure.clear();
        ASSERT(isNull());
    }

    template<typename Visitor>
    void visitAggregate(Visitor& visitor)
    {
        visitor.append(m_structure);
    }

private:
    unsigned possibleDefaultPropertyCount(VM&, JSObject* prototype);

    Allocator m_allocator; // Precomputed to make things easier for generated code.
    WriteBarrier<Structure> m_structure;
};

class ObjectAllocationProfile : public ObjectAllocationProfileBase<ObjectAllocationProfile> {
public:
    using Base = ObjectAllocationProfileBase<ObjectAllocationProfile>;

    ObjectAllocationProfile() = default;

    using Base::clear;
    using Base::visitAggregate;

    void setPrototype(VM&, JSCell*, JSObject*) { }
};

class ObjectAllocationProfileWithPrototype : public ObjectAllocationProfileBase<ObjectAllocationProfileWithPrototype> {
public:
    using Base = ObjectAllocationProfileBase<ObjectAllocationProfileWithPrototype>;

    static constexpr ptrdiff_t offsetOfPrototype() { return OBJECT_OFFSETOF(ObjectAllocationProfileWithPrototype, m_prototype); }

    ObjectAllocationProfileWithPrototype() = default;

    JSObject* prototype()
    {
        JSObject* prototype = m_prototype.get();
        WTF::dependentLoadLoadFence();
        return prototype;
    }

    void clear()
    {
        Base::clear();
        m_prototype.clear();
        ASSERT(isNull());
    }

    template<typename Visitor>
    void visitAggregate(Visitor& visitor)
    {
        Base::visitAggregate(visitor);
        visitor.append(m_prototype);
    }

    void setPrototype(VM& vm, JSCell* owner, JSObject* object)
    {
        m_prototype.set(vm, owner, object);
    }

private:
    WriteBarrier<JSObject> m_prototype;
};



} // namespace JSC
