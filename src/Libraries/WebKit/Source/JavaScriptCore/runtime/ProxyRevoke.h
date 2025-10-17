/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#include "InternalFunction.h"

namespace JSC {

class ProxyObject;

class ProxyRevoke final : public InternalFunction {
public:
    typedef InternalFunction Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.proxyRevokeSpace<mode>();
    }

    static ProxyRevoke* create(VM&, Structure*, ProxyObject*);

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    void finishCreation(VM&);
    DECLARE_VISIT_CHILDREN;
    JSValue proxy() { return m_proxy.get(); }
    void setProxyToNull(VM& vm) { return m_proxy.set(vm, this, jsNull()); }

private:
    ProxyRevoke(VM&, Structure*, ProxyObject*);

    WriteBarrier<Unknown> m_proxy;
};

} // namespace JSC
