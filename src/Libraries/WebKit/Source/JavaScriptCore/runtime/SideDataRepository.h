/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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

#include <wtf/HashMap.h>
#include <wtf/Locker.h>
#include <wtf/NeverDestroyed.h>

namespace JSC {

class SideDataRepository {
public:
    class SideData {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        virtual ~SideData() = default;
    };

    template<typename Type, typename Functor>
    Type& ensure(void* owner, void* key, const Functor& functor)
    {
        static_assert(std::is_base_of_v<SideData, Type>);
        Locker lock { m_lock };
        auto result = add(owner, key, nullptr);
        if (result.isNewEntry)
            result.iterator->value = functor();
        return *reinterpret_cast<Type*>(result.iterator->value.get());
    }

    void deleteAll(void* owner);

protected:
    using KeyValueStore = UncheckedKeyHashMap<void*, std::unique_ptr<SideData>>;
    using AddResult = KeyValueStore::AddResult;

    SideDataRepository() = default;

    JS_EXPORT_PRIVATE AddResult add(void* owner, void* key, std::unique_ptr<SideData>) WTF_REQUIRES_LOCK(m_lock);

    UncheckedKeyHashMap<void*, KeyValueStore> m_ownerStore WTF_GUARDED_BY_LOCK(m_lock);
    Lock m_lock;

    friend class LazyNeverDestroyed<SideDataRepository>;
};

JS_EXPORT_PRIVATE SideDataRepository& sideDataRepository();

} // namespace JSC
