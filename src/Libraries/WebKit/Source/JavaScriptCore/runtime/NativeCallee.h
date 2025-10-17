/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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

#include "CalleeBits.h"
#include "ImplementationVisibility.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace JSC {

class LLIntOffsetsExtractor;

class NativeCallee : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<NativeCallee> {
    WTF_MAKE_COMPACT_TZONE_ALLOCATED(NativeCallee);
public:
    enum class Category : uint8_t {
        InlineCache,
        Wasm,
    };

    Category category() const { return m_category; }
    ImplementationVisibility implementationVisibility() const { return m_implementationVisibility; }

    void dump(PrintStream&) const;

    JS_EXPORT_PRIVATE void operator delete(NativeCallee*, std::destroying_delete_t);

protected:
    JS_EXPORT_PRIVATE NativeCallee(Category, ImplementationVisibility);

private:
    Category m_category;
    ImplementationVisibility m_implementationVisibility { ImplementationVisibility::Public };
};

// This lets you do a RefPtr<NativeCallee, BoxedNativeCalleePtrTraits<NativeCallee>>
template<typename T>
class BoxedNativeCalleePtrTraits {
public:
    using StorageType = CalleeBits;
    // Use an intermediate cast to uintptr_t to silence unsafe casting warning. It's locally "obvious" (other than the fact that RefPtr uses StorageType's constructor instead of a wrap) the
    // return value is of type T.
    static T* unwrap(const CalleeBits& calleeBits) { return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(calleeBits.asNativeCallee())); }
    static T* exchange(CalleeBits& calleeBits, T* newCallee)
    {
        T* result = unwrap(calleeBits);
        calleeBits = newCallee;
        return result;
    }

    // FIXME: This isn't hashable since we don't have hashTableDeletedValue() or isHashTableDeletedValue()
    // but those probably shouldn't be hard to add if needed.
};

} // namespace JSC
