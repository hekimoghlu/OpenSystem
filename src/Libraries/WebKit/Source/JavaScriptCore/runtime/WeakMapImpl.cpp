/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "config.h"
#include "WeakMapImpl.h"

#include "AuxiliaryBarrierInlines.h"
#include "SlotVisitorInlines.h"
#include "StructureInlines.h"
#include "WeakMapImplInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

template <typename WeakMapBucket>
void WeakMapImpl<WeakMapBucket>::destroy(JSCell* cell)
{
    static_cast<WeakMapImpl*>(cell)->~WeakMapImpl();
}

template<typename WeakMapBucket>
template<typename Visitor>
void WeakMapImpl<WeakMapBucket>::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    WeakMapImpl* thisObject = jsCast<WeakMapImpl*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.reportExtraMemoryVisited(thisObject->m_capacity * sizeof(WeakMapBucket));
}

DEFINE_VISIT_CHILDREN_WITH_MODIFIER(template<typename WeakMapBucket>, WeakMapImpl<WeakMapBucket>);

template <typename WeakMapBucket>
size_t WeakMapImpl<WeakMapBucket>::estimatedSize(JSCell* cell, VM& vm)
{
    auto* thisObject = static_cast<WeakMapImpl*>(cell);
    return Base::estimatedSize(thisObject, vm) + (sizeof(WeakMapImpl) - sizeof(Base)) + thisObject->m_capacity * sizeof(WeakMapBucket);
}

template <>
template <>
void WeakMapImpl<WeakMapBucket<WeakMapBucketDataKey>>::visitOutputConstraints(JSCell*, AbstractSlotVisitor&)
{
    // Only JSWeakMap needs to harvest value references
}

template <>
template <>
void WeakMapImpl<WeakMapBucket<WeakMapBucketDataKey>>::visitOutputConstraints(JSCell*, SlotVisitor&)
{
    // Only JSWeakMap needs to harvest value references
}

template<typename BucketType>
template<typename Visitor>
ALWAYS_INLINE void WeakMapImpl<BucketType>::visitOutputConstraints(JSCell* cell, Visitor& visitor)
{
    static_assert(std::is_same<BucketType, WeakMapBucket<WeakMapBucketDataKeyValue>>::value);

    auto* thisObject = jsCast<WeakMapImpl*>(cell);
    auto* buffer = thisObject->buffer();
    for (uint32_t index = 0; index < thisObject->m_capacity; ++index) {
        auto* bucket = buffer + index;
        if (bucket->isEmpty() || bucket->isDeleted())
            continue;
        if (!visitor.isMarked(bucket->key()))
            continue;
        bucket->visitAggregate(visitor);
    }
}

template void WeakMapImpl<WeakMapBucket<WeakMapBucketDataKeyValue>>::visitOutputConstraints(JSCell*, AbstractSlotVisitor&);
template void WeakMapImpl<WeakMapBucket<WeakMapBucketDataKeyValue>>::visitOutputConstraints(JSCell*, SlotVisitor&);

template <typename WeakMapBucket>
template<typename Appender>
void WeakMapImpl<WeakMapBucket>::takeSnapshotInternal(unsigned limit, Appender appender)
{
    DisallowGC disallowGC;
    unsigned fetched = 0;
    forEach([&](JSCell* key, JSValue value) {
        appender(key, value);
        ++fetched;
        if (limit && fetched >= limit)
            return IterationState::Stop;
        return IterationState::Continue;
    });
}

// takeSnapshot must not invoke garbage collection since iterating WeakMap may be modified.
template <>
void WeakMapImpl<WeakMapBucket<WeakMapBucketDataKey>>::takeSnapshot(MarkedArgumentBuffer& buffer, unsigned limit)
{
    takeSnapshotInternal(limit, [&](JSCell* key, JSValue) {
        buffer.append(key);
    });
}

template <>
void WeakMapImpl<WeakMapBucket<WeakMapBucketDataKeyValue>>::takeSnapshot(MarkedArgumentBuffer& buffer, unsigned limit)
{
    takeSnapshotInternal(limit, [&](JSCell* key, JSValue value) {
        buffer.append(key);
        buffer.append(value);
    });
}

template class WeakMapImpl<WeakMapBucket<WeakMapBucketDataKeyValue>>;
template class WeakMapImpl<WeakMapBucket<WeakMapBucketDataKey>>;

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
