/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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

class ProxyableAccessCase : public AccessCase {
public:
    using Base = AccessCase;
    friend class AccessCase;
    friend class InlineCacheCompiler;

    static Ref<AccessCase> create(VM&, JSCell*, AccessType, CacheableIdentifier, PropertyOffset, Structure*, const ObjectPropertyConditionSet& = ObjectPropertyConditionSet(),
        bool viaGlobalProxy = false, WatchpointSet* additionalSet = nullptr, RefPtr<PolyProtoAccessChain>&& = nullptr);

protected:
    ProxyableAccessCase(VM&, JSCell*, AccessType, CacheableIdentifier, PropertyOffset, Structure*, const ObjectPropertyConditionSet&, bool viaGlobalProxy, WatchpointSet* additionalSet, RefPtr<PolyProtoAccessChain>&&);

    WatchpointSet* additionalSetImpl() const { return m_additionalSet.get(); }
    void dumpImpl(PrintStream&, CommaPrinter&, Indenter&) const;

private:
    RefPtr<WatchpointSet> m_additionalSet;
};

} // namespace JSC

#endif // ENABLE(JIT)
