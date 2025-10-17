/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
#include "ProxyableAccessCase.h"

#if ENABLE(JIT)

namespace JSC {

ProxyableAccessCase::ProxyableAccessCase(VM& vm, JSCell* owner, AccessType accessType, CacheableIdentifier identifier, PropertyOffset offset, Structure* structure,
    const ObjectPropertyConditionSet& conditionSet, bool viaGlobalProxy, WatchpointSet* additionalSet, RefPtr<PolyProtoAccessChain>&& prototypeAccessChain)
    : Base(vm, owner, accessType, identifier, offset, structure, conditionSet, WTFMove(prototypeAccessChain))
    , m_additionalSet(additionalSet)
{
    m_viaGlobalProxy = viaGlobalProxy;
}

Ref<AccessCase> ProxyableAccessCase::create(VM& vm, JSCell* owner, AccessType type, CacheableIdentifier identifier, PropertyOffset offset, Structure* structure, const ObjectPropertyConditionSet& conditionSet, bool viaGlobalProxy, WatchpointSet* additionalSet, RefPtr<PolyProtoAccessChain>&& prototypeAccessChain)
{
    ASSERT(type == Load || type == Miss || type == GetGetter);
    return adoptRef(*new ProxyableAccessCase(vm, owner, type, identifier, offset, structure, conditionSet, viaGlobalProxy, additionalSet, WTFMove(prototypeAccessChain)));
}

void ProxyableAccessCase::dumpImpl(PrintStream& out, CommaPrinter& comma, Indenter& indent) const
{
    Base::dumpImpl(out, comma, indent);
    out.print(comma, "viaGlobalProxy = ", viaGlobalProxy());
    out.print(comma, "additionalSet = ", RawPointer(additionalSet()));
}

} // namespace JSC

#endif // ENABLE(JIT)
