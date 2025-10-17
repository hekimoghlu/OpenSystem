/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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

#include "PCToCodeOriginMap.h"
#include <wtf/Box.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class NativeCallee;

class NativeCalleeRegistry {
    WTF_MAKE_TZONE_ALLOCATED(NativeCalleeRegistry);
    WTF_MAKE_NONCOPYABLE(NativeCalleeRegistry);
public:
    static void initialize();
    static NativeCalleeRegistry& singleton();

    Lock& getLock() WTF_RETURNS_LOCK(m_lock) { return m_lock; }

    void registerCallee(NativeCallee* callee)
    {
        Locker locker { m_lock };
        auto addResult = m_calleeSet.add(callee);
        ASSERT_UNUSED(addResult, addResult.isNewEntry);
    }

    void unregisterCallee(NativeCallee* callee)
    {
        Locker locker { m_lock };
        m_calleeSet.remove(callee);
#if ENABLE(JIT)
        m_pcToCodeOriginMaps.remove(callee);
#endif
    }

    const UncheckedKeyHashSet<NativeCallee*>& allCallees() WTF_REQUIRES_LOCK(m_lock)
    {
        return m_calleeSet;
    }

    bool isValidCallee(NativeCallee* callee)  WTF_REQUIRES_LOCK(m_lock)
    {
        if (!UncheckedKeyHashSet<NativeCallee*>::isValidValue(callee))
            return false;
        return m_calleeSet.contains(callee);
    }

#if ENABLE(JIT)
    void addPCToCodeOriginMap(NativeCallee* callee, Box<PCToCodeOriginMap> originMap)
    {
        Locker locker { m_lock };
        ASSERT(isValidCallee(callee));
        auto addResult = m_pcToCodeOriginMaps.add(callee, WTFMove(originMap));
        RELEASE_ASSERT(addResult.isNewEntry);
    }

    Box<PCToCodeOriginMap> codeOriginMap(NativeCallee* callee)  WTF_REQUIRES_LOCK(m_lock)
    {
        ASSERT(isValidCallee(callee));
        auto iter = m_pcToCodeOriginMaps.find(callee);
        if (iter != m_pcToCodeOriginMaps.end())
            return iter->value;
        return nullptr;
    }
#endif

    NativeCalleeRegistry() = default;

private:
    Lock m_lock;
    UncheckedKeyHashSet<NativeCallee*> m_calleeSet WTF_GUARDED_BY_LOCK(m_lock);
#if ENABLE(JIT)
    UncheckedKeyHashMap<NativeCallee*, Box<PCToCodeOriginMap>> m_pcToCodeOriginMaps WTF_GUARDED_BY_LOCK(m_lock);
#endif
};

} // namespace JSC
