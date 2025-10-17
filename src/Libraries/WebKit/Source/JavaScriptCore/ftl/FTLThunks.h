/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#include "FTLLocation.h"
#include "FTLSlowPathCallKey.h"
#include "MacroAssemblerCodeRef.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class VM;

namespace FTL {

MacroAssemblerCodeRef<JITThunkPtrTag> osrExitGenerationThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> lazySlowPathGenerationThunkGenerator(VM&);
MacroAssemblerCodeRef<JITThunkPtrTag> slowPathCallThunkGenerator(VM&, const SlowPathCallKey&);

template<typename KeyTypeArgument>
struct ThunkMap {
    typedef KeyTypeArgument KeyType;
    typedef UncheckedKeyHashMap<KeyType, MacroAssemblerCodeRef<JITThunkPtrTag>> ToThunkMap;
    typedef UncheckedKeyHashMap<CodePtr<JITThunkPtrTag>, KeyType> FromThunkMap;
    
    ToThunkMap m_toThunk;
    FromThunkMap m_fromThunk;
};

template<typename MapType, typename GeneratorType>
MacroAssemblerCodeRef<JITThunkPtrTag> generateIfNecessary(
    VM& vm, MapType& map, const typename MapType::KeyType& key, GeneratorType generator)
{
    typename MapType::ToThunkMap::iterator iter = map.m_toThunk.find(key);
    if (iter != map.m_toThunk.end())
        return iter->value;

    MacroAssemblerCodeRef<JITThunkPtrTag> result = generator(vm, key);
    map.m_toThunk.add(key, result);
    map.m_fromThunk.add(result.code(), key);
    return result;
}

template<typename MapType>
typename MapType::KeyType keyForThunk(MapType& map, CodePtr<JITThunkPtrTag> ptr)
{
    typename MapType::FromThunkMap::iterator iter = map.m_fromThunk.find(ptr);
    RELEASE_ASSERT(iter != map.m_fromThunk.end());
    return iter->value;
}

class Thunks {
    WTF_MAKE_TZONE_ALLOCATED(Thunks);
    WTF_MAKE_NONCOPYABLE(Thunks);
public:
    Thunks() = default;
    MacroAssemblerCodeRef<JITThunkPtrTag> getSlowPathCallThunk(VM& vm, const SlowPathCallKey& key)
    {
        Locker locker { m_lock };
        return generateIfNecessary(vm, m_slowPathCallThunks, key, slowPathCallThunkGenerator);
    }

    SlowPathCallKey keyForSlowPathCallThunk(CodePtr<JITThunkPtrTag> ptr)
    {
        Locker locker { m_lock };
        return keyForThunk(m_slowPathCallThunks, ptr);
    }
    
private:
    Lock m_lock;
    ThunkMap<SlowPathCallKey> m_slowPathCallThunks;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
