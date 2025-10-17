/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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

#include "Identifier.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class ModuleScopeData : public RefCounted<ModuleScopeData> {
    WTF_MAKE_NONCOPYABLE(ModuleScopeData);
    WTF_MAKE_TZONE_ALLOCATED(ModuleScopeData);
public:
    typedef UncheckedKeyHashMap<RefPtr<UniquedStringImpl>, Vector<RefPtr<UniquedStringImpl>>, IdentifierRepHash, HashTraits<RefPtr<UniquedStringImpl>>> IdentifierAliasMap;

    static Ref<ModuleScopeData> create() { return adoptRef(*new ModuleScopeData); }

    const IdentifierAliasMap& exportedBindings() const { return m_exportedBindings; }

    bool exportName(const Identifier& exportedName)
    {
        return m_exportedNames.add(exportedName.impl()).isNewEntry;
    }

    void exportBinding(const Identifier& localName, const Identifier& exportedName)
    {
        m_exportedBindings.add(localName.impl(), Vector<RefPtr<UniquedStringImpl>>()).iterator->value.append(exportedName.impl());
    }

    void exportBinding(const Identifier& localName)
    {
        exportBinding(localName, localName);
    }

private:
    ModuleScopeData() = default;

    IdentifierSet m_exportedNames { };
    IdentifierAliasMap m_exportedBindings { };
};

} // namespace
