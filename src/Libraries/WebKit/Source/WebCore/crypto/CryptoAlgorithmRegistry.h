/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

#include "CryptoAlgorithmIdentifier.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class CryptoAlgorithm;

class CryptoAlgorithmRegistry {
    WTF_MAKE_NONCOPYABLE(CryptoAlgorithmRegistry);
    friend class LazyNeverDestroyed<CryptoAlgorithmRegistry>;

public:
    static CryptoAlgorithmRegistry& singleton();

    std::optional<CryptoAlgorithmIdentifier> identifier(const String&);
    String name(CryptoAlgorithmIdentifier);

    RefPtr<CryptoAlgorithm> create(CryptoAlgorithmIdentifier);

private:
    CryptoAlgorithmRegistry();
    void platformRegisterAlgorithms();

    using CryptoAlgorithmConstructor = Ref<CryptoAlgorithm> (*)();

    template<typename AlgorithmClass> void registerAlgorithm()
    {
        registerAlgorithm(AlgorithmClass::s_name, AlgorithmClass::s_identifier, AlgorithmClass::create);
    }

    void registerAlgorithm(const String& name, CryptoAlgorithmIdentifier, CryptoAlgorithmConstructor);

    Lock m_lock;
    HashMap<String, CryptoAlgorithmIdentifier, ASCIICaseInsensitiveHash> m_identifiers WTF_GUARDED_BY_LOCK(m_lock);
    HashMap<unsigned, std::pair<String, CryptoAlgorithmConstructor>> m_constructors WTF_GUARDED_BY_LOCK(m_lock);
};

} // namespace WebCore
