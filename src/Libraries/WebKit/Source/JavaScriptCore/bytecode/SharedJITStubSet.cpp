/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#include "SharedJITStubSet.h"

#include "BaselineJITRegisters.h"
#include "CacheableIdentifierInlines.h"
#include "DFGJITCode.h"
#include "InlineCacheCompiler.h"
#include "Repatch.h"

namespace JSC {

#if ENABLE(JIT)

RefPtr<PolymorphicAccessJITStubRoutine> SharedJITStubSet::getStatelessStub(StatelessCacheKey key) const
{
    return m_statelessStubs.get(key);
}

void SharedJITStubSet::setStatelessStub(StatelessCacheKey key, Ref<PolymorphicAccessJITStubRoutine> stub)
{
    m_statelessStubs.add(key, WTFMove(stub));
}

MacroAssemblerCodeRef<JITStubRoutinePtrTag> SharedJITStubSet::getDOMJITCode(DOMJITCacheKey key) const
{
    return m_domJITCodes.get(key);
}

void SharedJITStubSet::setDOMJITCode(DOMJITCacheKey key, MacroAssemblerCodeRef<JITStubRoutinePtrTag> code)
{
    m_domJITCodes.add(key, WTFMove(code));
}

RefPtr<InlineCacheHandler> SharedJITStubSet::getSlowPathHandler(AccessType type) const
{
    return m_slowPathHandlers[static_cast<unsigned>(type)];
}

void SharedJITStubSet::setSlowPathHandler(AccessType type, Ref<InlineCacheHandler> handler)
{
    m_slowPathHandlers[static_cast<unsigned>(type)] = WTFMove(handler);
}

#endif // ENABLE(JIT)

} // namespace JSC
