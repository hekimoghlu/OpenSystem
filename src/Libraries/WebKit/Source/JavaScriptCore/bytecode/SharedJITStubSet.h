/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#include "DOMJITGetterSetter.h"
#include "StructureStubInfo.h"

namespace JSC {

#if ENABLE(JIT)

class SharedJITStubSet {
    WTF_MAKE_FAST_ALLOCATED(SharedJITStubSet);
public:
    SharedJITStubSet() = default;

    using StructureStubInfoKey = std::tuple<AccessType, bool, bool, bool, bool>;
    using StatelessCacheKey = std::tuple<StructureStubInfoKey, AccessCase::AccessType>;
    using DOMJITCacheKey = std::tuple<StructureStubInfoKey, const DOMJIT::GetterSetter*>;

    static StructureStubInfoKey stubInfoKey(const StructureStubInfo& stubInfo)
    {
        return std::tuple { stubInfo.accessType, static_cast<bool>(stubInfo.propertyIsInt32), static_cast<bool>(stubInfo.propertyIsString), static_cast<bool>(stubInfo.propertyIsSymbol), static_cast<bool>(stubInfo.prototypeIsKnownObject) };
    }

    struct Hash {
        struct Key {
            Key() = default;

            Key(StructureStubInfoKey stubInfoKey, PolymorphicAccessJITStubRoutine* wrapped)
                : m_wrapped(wrapped)
                , m_stubInfoKey(stubInfoKey)
            { }

            Key(WTF::HashTableDeletedValueType)
                : m_wrapped(std::bit_cast<PolymorphicAccessJITStubRoutine*>(static_cast<uintptr_t>(1)))
            { }

            bool isHashTableDeletedValue() const { return m_wrapped == std::bit_cast<PolymorphicAccessJITStubRoutine*>(static_cast<uintptr_t>(1)); }

            friend bool operator==(const Key&, const Key&) = default;

            PolymorphicAccessJITStubRoutine* m_wrapped { nullptr };
            StructureStubInfoKey m_stubInfoKey { };
        };

        using KeyTraits = SimpleClassHashTraits<Key>;

        static unsigned hash(const Key& p)
        {
            if (!p.m_wrapped)
                return 1;
            return p.m_wrapped->hash();
        }

        static bool equal(const Key& a, const Key& b)
        {
            return a == b;
        }

        static constexpr bool safeToCompareToEmptyOrDeleted = false;
    };

    struct Searcher {
        struct Translator {
            static unsigned hash(const Searcher& searcher)
            {
                return searcher.m_hash;
            }

            static bool equal(const Hash::Key a, const Searcher& b)
            {
                if (a.m_stubInfoKey == b.m_stubInfoKey && Hash::hash(a) == b.m_hash) {
                    if (a.m_wrapped->cases().size() != 1)
                        return false;
                    const auto& aCase = a.m_wrapped->cases()[0];
                    const auto& bCase = b.m_accessCase;
                    if (!AccessCase::canBeShared(aCase.get(), bCase.get()))
                        return false;
                    return true;
                }
                return false;
            }
        };

        Searcher(StructureStubInfoKey&& stubInfoKey, Ref<AccessCase>&& accessCase)
            : m_stubInfoKey(WTFMove(stubInfoKey))
            , m_accessCase(WTFMove(accessCase))
            , m_hash(m_accessCase->hash())
        {
        }

        StructureStubInfoKey m_stubInfoKey;
        Ref<AccessCase> m_accessCase;
        unsigned m_hash { 0 };
    };

    struct PointerTranslator {
        static unsigned hash(const PolymorphicAccessJITStubRoutine* stub)
        {
            return stub->hash();
        }

        static bool equal(const Hash::Key& key, const PolymorphicAccessJITStubRoutine* stub)
        {
            return key.m_wrapped == stub;
        }
    };

    void add(Hash::Key&& key)
    {
        m_stubs.add(WTFMove(key));
    }

    void remove(PolymorphicAccessJITStubRoutine* stub)
    {
        auto iter = m_stubs.find<PointerTranslator>(stub);
        if (iter != m_stubs.end())
            m_stubs.remove(iter);
    }

    RefPtr<PolymorphicAccessJITStubRoutine> find(const Searcher& searcher)
    {
        auto entry = m_stubs.find<SharedJITStubSet::Searcher::Translator>(searcher);
        if (entry != m_stubs.end())
            return entry->m_wrapped;
        return nullptr;
    }

    RefPtr<PolymorphicAccessJITStubRoutine> getStatelessStub(StatelessCacheKey) const;
    void setStatelessStub(StatelessCacheKey, Ref<PolymorphicAccessJITStubRoutine>);

    MacroAssemblerCodeRef<JITStubRoutinePtrTag> getDOMJITCode(DOMJITCacheKey) const;
    void setDOMJITCode(DOMJITCacheKey, MacroAssemblerCodeRef<JITStubRoutinePtrTag>);

    RefPtr<InlineCacheHandler> getSlowPathHandler(AccessType) const;
    void setSlowPathHandler(AccessType, Ref<InlineCacheHandler>);

private:
    UncheckedKeyHashSet<Hash::Key, Hash, Hash::KeyTraits> m_stubs;
    UncheckedKeyHashMap<StatelessCacheKey, Ref<PolymorphicAccessJITStubRoutine>> m_statelessStubs;
    UncheckedKeyHashMap<DOMJITCacheKey, MacroAssemblerCodeRef<JITStubRoutinePtrTag>> m_domJITCodes;
    std::array<RefPtr<InlineCacheHandler>, numberOfAccessTypes> m_fallbackHandlers { };
    std::array<RefPtr<InlineCacheHandler>, numberOfAccessTypes> m_slowPathHandlers { };
};

#else

class StructureStubInfo;

#endif // ENABLE(JIT)

} // namespace JSC
