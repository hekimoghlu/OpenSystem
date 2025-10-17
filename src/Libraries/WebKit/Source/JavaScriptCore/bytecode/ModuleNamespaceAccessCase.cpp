/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#include "ModuleNamespaceAccessCase.h"

#if ENABLE(JIT)

#include "CCallHelpers.h"
#include "InlineCacheCompiler.h"
#include "JSModuleEnvironment.h"
#include "JSModuleNamespaceObject.h"
#include "StructureStubInfo.h"

namespace JSC {

ModuleNamespaceAccessCase::ModuleNamespaceAccessCase(VM& vm, JSCell* owner, CacheableIdentifier identifier, JSModuleNamespaceObject* moduleNamespaceObject, JSModuleEnvironment* moduleEnvironment, ScopeOffset scopeOffset)
    : Base(vm, owner, AccessType::ModuleNamespaceLoad, identifier, invalidOffset, nullptr, ObjectPropertyConditionSet(), nullptr)
    , m_scopeOffset(scopeOffset)
{
    m_moduleNamespaceObject.set(vm, owner, moduleNamespaceObject);
    m_moduleEnvironment.set(vm, owner, moduleEnvironment);
}

Ref<AccessCase> ModuleNamespaceAccessCase::create(VM& vm, JSCell* owner, CacheableIdentifier identifier, JSModuleNamespaceObject* moduleNamespaceObject, JSModuleEnvironment* moduleEnvironment, ScopeOffset scopeOffset)
{
    return adoptRef(*new ModuleNamespaceAccessCase(vm, owner, identifier, moduleNamespaceObject, moduleEnvironment, scopeOffset));
}

} // namespace JSC

#endif // ENABLE(JIT)
