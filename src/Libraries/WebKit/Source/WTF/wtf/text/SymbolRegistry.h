/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

#include <wtf/HashSet.h>
#include <wtf/text/StringHash.h>

namespace WTF {

class RegisteredSymbolImpl;

class SymbolRegistry {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(SymbolRegistry);
public:
    enum class Type : uint8_t { PublicSymbol, PrivateSymbol };
    WTF_EXPORT_PRIVATE SymbolRegistry(Type = Type::PublicSymbol);
    WTF_EXPORT_PRIVATE ~SymbolRegistry();

    WTF_EXPORT_PRIVATE Ref<RegisteredSymbolImpl> symbolForKey(const String&);

    void remove(RegisteredSymbolImpl&);

private:
    UncheckedKeyHashSet<RefPtr<StringImpl>> m_table;
    Type m_symbolType;
};

}
