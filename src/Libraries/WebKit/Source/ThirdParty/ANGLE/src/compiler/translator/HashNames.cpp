/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/HashNames.h"

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"

namespace sh
{

namespace
{
constexpr const ImmutableString kHashedNamePrefix("webgl_");

ImmutableString HashName(const ImmutableString &name, ShHashFunction64 hashFunction)
{
    ASSERT(!name.empty());
    ASSERT(hashFunction);
    khronos_uint64_t number = (*hashFunction)(name.data(), name.length());

    // Build the hashed name in place.
    static const unsigned int kHexStrMaxLength = sizeof(number) * 2;
    static const size_t kHashedNameMaxLength   = kHashedNamePrefix.length() + kHexStrMaxLength;

    ImmutableStringBuilder hashedName(kHashedNameMaxLength);
    hashedName << kHashedNamePrefix;

    hashedName.appendHex(number);

    return hashedName;
}

void AddToNameMapIfNotMapped(const ImmutableString &name,
                             const ImmutableString &hashedName,
                             NameMap *nameMap)
{
    if (nameMap)
    {
        NameMap::const_iterator it = nameMap->find(name.data());
        if (it != nameMap->end())
        {
            // (How bout returning?)
            return;
        }
        (*nameMap)[name.data()] = hashedName.data();
    }
}

}  // anonymous namespace

ImmutableString HashName(const ImmutableString &name,
                         ShHashFunction64 hashFunction,
                         NameMap *nameMap)
{
    const ImmutableString kUnhashedNamePrefix(kUserDefinedNamePrefix);

    if (hashFunction == nullptr)
    {
        if (name.length() + kUnhashedNamePrefix.length() > kESSLMaxIdentifierLength)
        {
            // If the identifier length is already close to the limit, we can't prefix it. This is
            // not a problem since there are no builtins or ANGLE's internal variables that would
            // have as long names and could conflict.
            return name;
        }
        ImmutableStringBuilder prefixedName(kUnhashedNamePrefix.length() + name.length());
        prefixedName << kUnhashedNamePrefix << name;
        ImmutableString res = prefixedName;
        AddToNameMapIfNotMapped(name, res, nameMap);
        return res;
    }

    // Has a hash function
    ImmutableString hashedName = HashName(name, hashFunction);
    AddToNameMapIfNotMapped(name, hashedName, nameMap);
    return hashedName;
}

ImmutableString HashName(const TSymbol *symbol, ShHashFunction64 hashFunction, NameMap *nameMap)
{
    if (symbol->symbolType() == SymbolType::Empty)
    {
        return kEmptyImmutableString;
    }
    if (symbol->symbolType() == SymbolType::AngleInternal ||
        symbol->symbolType() == SymbolType::BuiltIn)
    {
        return symbol->name();
    }
    return HashName(symbol->name(), hashFunction, nameMap);
}

}  // namespace sh
