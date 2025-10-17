/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

#include "DirectEvalExecutable.h"
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/text/StringHash.h>

namespace JSC {

    class SlotVisitor;

    class DirectEvalCodeCache {
    public:
        class CacheKey {
        public:
            CacheKey(const String& source, BytecodeIndex bytecodeIndex)
                : m_source(source.impl())
                , m_bytecodeIndex(bytecodeIndex)
            {
            }

            CacheKey(WTF::HashTableDeletedValueType)
                : m_source(WTF::HashTableDeletedValue)
            {
            }

            CacheKey() = default;

            unsigned hash() const { return m_source->hash() ^ m_bytecodeIndex.asBits(); }

            bool isEmptyValue() const { return !m_source; }

            bool operator==(const CacheKey& other) const
            {
                return m_bytecodeIndex == other.m_bytecodeIndex && WTF::equal(m_source.get(), other.m_source.get());
            }

            bool isHashTableDeletedValue() const { return m_source.isHashTableDeletedValue(); }

            struct Hash {
                static unsigned hash(const CacheKey& key)
                {
                    return key.hash();
                }
                static bool equal(const CacheKey& lhs, const CacheKey& rhs)
                {
                    return lhs == rhs;
                }
                static constexpr bool safeToCompareToEmptyOrDeleted = false;
            };

            typedef SimpleClassHashTraits<CacheKey> HashTraits;

        private:
            RefPtr<StringImpl> m_source;
            BytecodeIndex m_bytecodeIndex;
        };

        DirectEvalExecutable* tryGet(const String& evalSource, BytecodeIndex bytecodeIndex)
        {
            return m_cacheMap.inlineGet(CacheKey(evalSource, bytecodeIndex)).get();
        }
        
        void set(JSGlobalObject* globalObject, JSCell* owner, const String& evalSource, BytecodeIndex bytecodeIndex, DirectEvalExecutable* evalExecutable)
        {
            if (m_cacheMap.size() < maxCacheEntries)
                setSlow(globalObject, owner, evalSource, bytecodeIndex, evalExecutable);
        }

        bool isEmpty() const { return m_cacheMap.isEmpty(); }

        DECLARE_VISIT_AGGREGATE;

        void clear();

    private:
        static constexpr int maxCacheEntries = 64;

        void setSlow(JSGlobalObject*, JSCell* owner, const String& evalSource, BytecodeIndex, DirectEvalExecutable*);

        typedef UncheckedKeyHashMap<CacheKey, WriteBarrier<DirectEvalExecutable>, CacheKey::Hash, CacheKey::HashTraits> EvalCacheMap;
        EvalCacheMap m_cacheMap;
        Lock m_lock;
    };

} // namespace JSC
