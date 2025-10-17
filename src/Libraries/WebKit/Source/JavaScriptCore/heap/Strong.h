/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include "Handle.h"
#include "HandleSet.h"
#include "Heap.h"
#include "JSLock.h"
#include "StrongForward.h"
#include <wtf/RefTrackerMixin.h>

namespace JSC {

class VM;

#if ENABLE(REFTRACKER)
void initializeSystemForStrongRefTracker();
#endif

REFTRACKER_DECL(StrongRefTracker, {
    initializeSystemForStrongRefTracker();
});

// A strongly referenced handle that prevents the object it points to from being garbage collected.
template <typename T, ShouldStrongDestructorGrabLock shouldStrongDestructorGrabLock> class Strong final : public Handle<T> {
    using Handle<T>::slot;
    using Handle<T>::setSlot;
    template <typename U, ShouldStrongDestructorGrabLock> friend class Strong;

public:
    typedef typename Handle<T>::ExternalType ExternalType;

    Strong()
        : Handle<T>()
    {
    }

    inline Strong(VM&, ExternalType = ExternalType());

    inline Strong(VM&, Handle<T>);

    Strong(const Strong& other)
        : Handle<T>()
    {
        if (!other.slot())
            return;
        setSlot(HandleSet::heapFor(other.slot())->allocate());
        set(other.get());
    }

    template <typename U> Strong(const Strong<U>& other)
        : Handle<T>()
    {
        if (!other.slot())
            return;
        setSlot(HandleSet::heapFor(other.slot())->allocate());
        set(other.get());
    }

    enum HashTableDeletedValueTag { HashTableDeletedValue };
    bool isHashTableDeletedValue() const { return slot() == hashTableDeletedValue(); }

    Strong(HashTableDeletedValueTag)
        : Handle<T>(hashTableDeletedValue())
    {
    }

    enum HashTableEmptyValueTag { HashTableEmptyValue };
    bool isHashTableEmptyValue() const { return slot() == hashTableEmptyValue(); }

    Strong(HashTableEmptyValueTag)
        : Handle<T>(hashTableEmptyValue())
    {
    }

    ~Strong() override
    {
        clear();
    }

    bool operator!() const { return !slot() || !*slot(); }

    explicit operator bool() const { return !!*this; }

    void swap(Strong& other)
    {
        Handle<T>::swap(other);
    }

    ExternalType get() const { return HandleTypes<T>::getFromSlot(this->slot()); }

    inline void set(VM&, ExternalType);

    template <typename U> Strong& operator=(const Strong<U>& other)
    {
        if (!other.slot()) {
            clear();
            return *this;
        }

        set(*HandleSet::heapFor(other.slot())->vm(), other.get());
        return *this;
    }

    Strong& operator=(const Strong& other)
    {
        if (!other.slot()) {
            clear();
            return *this;
        }

        set(HandleSet::heapFor(other.slot())->vm(), other.get());
        return *this;
    }

    void clear()
    {
        if (!slot())
            return;

        auto* heap = HandleSet::heapFor(slot());
        if (shouldStrongDestructorGrabLock == ShouldStrongDestructorGrabLock::Yes) {
            JSLockHolder holder(heap->vm());
            heap->deallocate(slot());
            setSlot(nullptr);
        } else {
            heap->deallocate(slot());
            setSlot(nullptr);
        }
    }

private:
    static HandleSlot hashTableDeletedValue() { return reinterpret_cast<HandleSlot>(-1); }
    static HandleSlot hashTableEmptyValue() { return reinterpret_cast<HandleSlot>(0); }

    void set(ExternalType externalType)
    {
        ASSERT(slot());
        JSValue value = HandleTypes<T>::toJSValue(externalType);
        HandleSet::heapFor(slot())->template writeBarrier<std::is_base_of_v<JSCell, T>>(slot(), value);
        *slot() = value;
    }

    REFTRACKER_MEMBERS(StrongRefTracker);
};

template<class T> inline void swap(Strong<T>& a, Strong<T>& b)
{
    a.swap(b);
}

} // namespace JSC

namespace WTF {

template<typename T> struct VectorTraits<JSC::Strong<T>> : SimpleClassVectorTraits {
    static constexpr bool canCompareWithMemcmp = false;
#if ENABLE(REFTRACKER)
    static constexpr bool canInitializeWithMemset = false;
    static constexpr bool canMoveWithMemcpy = false;
#endif
};

template<typename P> struct HashTraits<JSC::Strong<P>> : SimpleClassHashTraits<JSC::Strong<P>> {
#if ENABLE(REFTRACKER)
    using S = JSC::Strong<P>;
    static constexpr bool emptyValueIsZero = false;
    static S emptyValue() { return S::HashTableEmptyValue; }

    template <typename>
    static void constructEmptyValue(S& slot)
    {
        new (NotNull, std::addressof(slot)) S(S::HashTableEmptyValue);
    }

    static constexpr bool hasIsEmptyValueFunction = true;
    static bool isEmptyValue(const S& value) { return value.isHashTableEmptyValue(); }

    static void constructDeletedValue(S& slot) { new (NotNull, &slot) S(S::HashTableDeletedValue); }
    static bool isDeletedValue(const S& value) { return value.isHashTableDeletedValue(); }

#endif
};

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
