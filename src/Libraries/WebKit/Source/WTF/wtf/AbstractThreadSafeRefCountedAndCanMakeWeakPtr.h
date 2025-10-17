/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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

#include <wtf/AbstractRefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WTF {

class AbstractThreadSafeRefCountedAndCanMakeWeakPtr : public AbstractRefCounted {
public:
    virtual ThreadSafeWeakPtrControlBlock& controlBlock() const = 0;

private:
    template<typename> friend class ThreadSafeWeakHashSet;
    virtual size_t weakRefCount() const = 0;
};

}

using WTF::AbstractThreadSafeRefCountedAndCanMakeWeakPtr;

// Convinience macro that implements AbstractThreadSafeRefCountedAndCanMakeWeakPtr for you.
#define WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL \
    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); } \
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); } \
    ThreadSafeWeakPtrControlBlock& controlBlock() const final { return ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::controlBlock(); } \
    size_t weakRefCount() const final { return ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::weakRefCount(); } \
    using __Unused_type_for_semicolon = int

