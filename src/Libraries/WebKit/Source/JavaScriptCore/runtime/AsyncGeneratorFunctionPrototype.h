/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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

namespace JSC {

class AsyncGeneratorFunctionPrototype final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(AsyncGeneratorFunctionPrototype, Base);
        return &vm.plainObjectSpace();
    }

    DECLARE_INFO;

    static AsyncGeneratorFunctionPrototype* create(VM& vm, Structure* structure)
    {
        AsyncGeneratorFunctionPrototype* prototype = new (NotNull, allocateCell<AsyncGeneratorFunctionPrototype>(vm)) AsyncGeneratorFunctionPrototype(vm, structure);
        prototype->finishCreation(vm);
        return prototype;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    AsyncGeneratorFunctionPrototype(VM&, Structure*);
    void finishCreation(VM&);
};

} // namespace JSC
