/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#include "CryptoAlgorithmRegistry.h"

#include "CryptoAlgorithm.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

CryptoAlgorithmRegistry& CryptoAlgorithmRegistry::singleton()
{
    static LazyNeverDestroyed<CryptoAlgorithmRegistry> registry;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        registry.construct();
    });
    return registry;
}

CryptoAlgorithmRegistry::CryptoAlgorithmRegistry()
{
    platformRegisterAlgorithms();
}

std::optional<CryptoAlgorithmIdentifier> CryptoAlgorithmRegistry::identifier(const String& name)
{
    if (name.isEmpty())
        return std::nullopt;

    Locker locker { m_lock };

    // FIXME: How is it helpful to call isolatedCopy on the argument to find?
    auto identifier = m_identifiers.find(name.isolatedCopy());
    if (identifier == m_identifiers.end())
        return std::nullopt;

    return identifier->value;
}

String CryptoAlgorithmRegistry::name(CryptoAlgorithmIdentifier identifier)
{
    Locker locker { m_lock };

    auto contructor = m_constructors.find(enumToUnderlyingType(identifier));
    if (contructor == m_constructors.end())
        return { };

    return contructor->value.first.isolatedCopy();
}

RefPtr<CryptoAlgorithm> CryptoAlgorithmRegistry::create(CryptoAlgorithmIdentifier identifier)
{
    Locker locker { m_lock };

    auto contructor = m_constructors.find(enumToUnderlyingType(identifier));
    if (contructor == m_constructors.end())
        return nullptr;

    return contructor->value.second();
}

void CryptoAlgorithmRegistry::registerAlgorithm(const String& name, CryptoAlgorithmIdentifier identifier, CryptoAlgorithmConstructor constructor)
{
    Locker locker { m_lock };

    ASSERT(!m_identifiers.contains(name));
    ASSERT(!m_constructors.contains(enumToUnderlyingType(identifier)));

    m_identifiers.add(name, identifier);
    m_constructors.add(enumToUnderlyingType(identifier), std::make_pair(name, constructor));
}


} // namespace WebCore
