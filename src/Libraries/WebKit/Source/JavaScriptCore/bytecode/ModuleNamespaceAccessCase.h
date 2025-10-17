/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#if ENABLE(JIT)

#include "AccessCase.h"

namespace JSC {

class JSModuleEnvironment;
class JSModuleNamespaceObject;

class ModuleNamespaceAccessCase final : public AccessCase {
public:
    using Base = AccessCase;
    friend class AccessCase;
    friend class InlineCacheCompiler;

    JSModuleNamespaceObject* moduleNamespaceObject() const { return m_moduleNamespaceObject.get(); }
    JSModuleEnvironment* moduleEnvironment() const { return m_moduleEnvironment.get(); }
    ScopeOffset scopeOffset() const { return m_scopeOffset; }

    static Ref<AccessCase> create(VM&, JSCell* owner, CacheableIdentifier, JSModuleNamespaceObject*, JSModuleEnvironment*, ScopeOffset);

private:
    ModuleNamespaceAccessCase(VM&, JSCell* owner, CacheableIdentifier, JSModuleNamespaceObject*, JSModuleEnvironment*, ScopeOffset);

    WriteBarrier<JSModuleNamespaceObject> m_moduleNamespaceObject;
    WriteBarrier<JSModuleEnvironment> m_moduleEnvironment;
    ScopeOffset m_scopeOffset;
};

} // namespace JSC

#endif // ENABLE(JIT)
