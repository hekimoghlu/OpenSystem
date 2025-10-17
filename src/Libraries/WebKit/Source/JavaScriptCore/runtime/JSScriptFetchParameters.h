/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
#include "ScriptFetchParameters.h"
#include <wtf/Ref.h>

namespace JSC {

class JSScriptFetchParameters final : public JSCell {
public:
    using Base = JSCell;

    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    DECLARE_EXPORT_INFO;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.scriptFetchParametersSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSScriptFetchParameters* create(VM& vm, Structure* structure, Ref<ScriptFetchParameters>&& parameters)
    {
        auto* result = new (NotNull, allocateCell<JSScriptFetchParameters>(vm)) JSScriptFetchParameters(vm, structure, WTFMove(parameters));
        result->finishCreation(vm);
        return result;
    }

    static JSScriptFetchParameters* create(VM& vm, Ref<ScriptFetchParameters>&& parameters)
    {
        return create(vm, vm.scriptFetchParametersStructure.get(), WTFMove(parameters));
    }

    ScriptFetchParameters& parameters() const
    {
        return m_parameters.get();
    }

    static void destroy(JSCell*);

private:
    JSScriptFetchParameters(VM& vm, Structure* structure, Ref<ScriptFetchParameters>&& parameters)
        : Base(vm, structure)
        , m_parameters(WTFMove(parameters))
    {
    }

    Ref<ScriptFetchParameters> m_parameters;
};

} // namespace JSC
