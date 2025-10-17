/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

#if ENABLE(FTL_JIT)

#include "B3HeapRange.h"
#include "FTLAbbreviatedTypes.h"
#include "JSCJSValue.h"
#include <array>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>

namespace JSC { namespace FTL {

class AbstractHeapRepository;
class Output;
class TypedPointer;

class AbstractHeap {
    WTF_MAKE_NONCOPYABLE(AbstractHeap);
    WTF_MAKE_TZONE_ALLOCATED(AbstractHeap);
public:
    AbstractHeap()
    {
    }
    
    AbstractHeap(AbstractHeap* parent, const char* heapName, ptrdiff_t offset = 0);

    bool isInitialized() const { return !!m_heapName; }
    
    void initialize(AbstractHeap* parent, const char* heapName, ptrdiff_t offset = 0)
    {
        changeParent(parent);
        m_heapName = heapName;
        m_offset = offset;
    }
    
    void changeParent(AbstractHeap* parent);

    AbstractHeap* parent() const
    {
        ASSERT(isInitialized());
        return m_parent;
    }

    const Vector<AbstractHeap*>& children() const;
    
    const char* heapName() const
    {
        ASSERT(isInitialized());
        return m_heapName;
    }

    B3::HeapRange range() const
    {
        // This will not have a valid value until after all lowering is done. Do associate an
        // AbstractHeap with a B3::Value*, use AbstractHeapRepository::decorateXXX().
        if (!m_range)
            badRangeError();
        
        return m_range;
    }

    // WARNING: Not all abstract heaps have a meaningful offset.
    ptrdiff_t offset() const
    {
        ASSERT(isInitialized());
        return m_offset;
    }

    void compute(unsigned begin = 0);

    // Print information about just this heap.
    void shallowDump(PrintStream&) const;

    // Print information about this heap and its ancestors. This is the default.
    void dump(PrintStream&) const;

    // Print information about this heap and its descendants. This is a multi-line dump.
    void deepDump(PrintStream&, unsigned indent = 0) const;

private:
    friend class AbstractHeapRepository;

    NO_RETURN_DUE_TO_CRASH void badRangeError() const;

    AbstractHeap* m_parent { nullptr };
    Vector<AbstractHeap*> m_children;
    intptr_t m_offset { 0 };
    B3::HeapRange m_range;
    const char* m_heapName { nullptr };
};

class IndexedAbstractHeap {
public:
    IndexedAbstractHeap(AbstractHeap* parent, const char* heapName, ptrdiff_t offset, size_t elementSize);
    ~IndexedAbstractHeap();
    
    AbstractHeap& atAnyIndex() { return m_heapForAnyIndex; }
    
    const AbstractHeap& at(ptrdiff_t index)
    {
        if (static_cast<size_t>(index) < m_smallIndices.size())
            return returnInitialized(m_smallIndices[index], index);
        return atSlow(index);
    }
    
    const AbstractHeap& operator[](ptrdiff_t index) { return at(index); }
    
    TypedPointer baseIndex(Output& out, LValue base, LValue index, JSValue indexAsConstant = JSValue(), ptrdiff_t offset = 0, LValue mask = nullptr);
    
    void dump(PrintStream&);

private:
    const AbstractHeap& returnInitialized(AbstractHeap& field, ptrdiff_t index)
    {
        if (UNLIKELY(!field.isInitialized()))
            initialize(field, index);
        return field;
    }

    const AbstractHeap& atSlow(ptrdiff_t index);
    void initialize(AbstractHeap& field, ptrdiff_t index);

    AbstractHeap m_heapForAnyIndex;
    size_t m_heapNameLength;
    ptrdiff_t m_offset;
    size_t m_elementSize;
    std::array<AbstractHeap, 16> m_smallIndices;
    
    struct WithoutZeroOrOneHashTraits : HashTraits<ptrdiff_t> {
        static void constructDeletedValue(ptrdiff_t& slot) { slot = 1; }
        static bool isDeletedValue(ptrdiff_t value) { return value == 1; }
    };
    typedef UncheckedKeyHashMap<ptrdiff_t, std::unique_ptr<AbstractHeap>, WTF::IntHash<ptrdiff_t>, WithoutZeroOrOneHashTraits> MapType;
    
    std::unique_ptr<MapType> m_largeIndices;
    Vector<CString, 16> m_largeIndexNames;
};

// A numbered abstract heap is like an indexed abstract heap, except that you
// can't rely on there being a relationship between the number you use to
// retrieve the sub-heap, and the offset that this heap has. (In particular,
// the sub-heaps don't have indices.)

class NumberedAbstractHeap {
public:
    NumberedAbstractHeap(AbstractHeap* parent, const char* heapName);
    ~NumberedAbstractHeap();
    
    AbstractHeap& atAnyNumber() { return m_indexedHeap.atAnyIndex(); }
    
    const AbstractHeap& at(unsigned number) { return m_indexedHeap.at(number); }
    const AbstractHeap& operator[](unsigned number) { return at(number); }

    void dump(PrintStream&);

private:
    
    // We use the fact that the indexed heap already has a superset of the
    // functionality we need.
    IndexedAbstractHeap m_indexedHeap;
};

class AbsoluteAbstractHeap {
public:
    AbsoluteAbstractHeap(AbstractHeap* parent, const char* heapName);
    ~AbsoluteAbstractHeap();
    
    const AbstractHeap& atAnyAddress() { return m_indexedHeap.atAnyIndex(); }
    
    const AbstractHeap& at(const void* address)
    {
        return m_indexedHeap.at(std::bit_cast<ptrdiff_t>(address));
    }
    
    const AbstractHeap& operator[](const void* address) { return at(address); }

    void dump(PrintStream&);

private:
    // The trick here is that the indexed heap is "indexed" by a pointer-width
    // integer. Pointers are themselves pointer-width integers. So we can reuse
    // all of the functionality.
    IndexedAbstractHeap m_indexedHeap;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
