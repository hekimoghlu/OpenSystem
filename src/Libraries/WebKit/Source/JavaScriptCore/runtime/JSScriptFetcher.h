/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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

#include "JSObject.h"
#include "ScriptFetcher.h"
#include <wtf/RefPtr.h>

namespace JSC {

class JSScriptFetcher final : public JSCell {
public:
    using Base = JSCell;

    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    DECLARE_EXPORT_INFO;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.scriptFetcherSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSScriptFetcher* create(VM& vm, Structure* structure, RefPtr<ScriptFetcher>&& scriptFetcher)
    {
        auto* result = new (NotNull, allocateCell<JSScriptFetcher>(vm)) JSScriptFetcher(vm, structure, WTFMove(scriptFetcher));
        result->finishCreation(vm);
        return result;
    }

    static JSScriptFetcher* create(VM& vm, RefPtr<ScriptFetcher>&& scriptFetcher)
    {
        return create(vm, vm.scriptFetcherStructure.get(), WTFMove(scriptFetcher));
    }

    ScriptFetcher* fetcher() const
    {
        return m_fetcher.get();
    }

    static void destroy(JSCell*);

private:
    JSScriptFetcher(VM& vm, Structure* structure, RefPtr<ScriptFetcher>&& scriptFetcher)
        : Base(vm, structure)
        , m_fetcher(WTFMove(scriptFetcher))
    {
    }

    RefPtr<ScriptFetcher> m_fetcher;
};

} // namespace JSC
