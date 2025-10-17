/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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

#include "JSPromiseConstructor.h"

namespace JSC {

class JSInternalPromise;
class JSInternalPromisePrototype;

class JSInternalPromiseConstructor final : public JSPromiseConstructor {
public:
    using Base = JSPromiseConstructor;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static JSInternalPromiseConstructor* create(VM&, Structure*, JSInternalPromisePrototype*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    JSInternalPromiseConstructor(VM&, FunctionExecutable*, JSGlobalObject*, Structure*);
};
static_assert(sizeof(JSInternalPromiseConstructor) == sizeof(JSFunction), "Allocate JSInternalPromiseConstructor in JSFunction IsoSubspace");

} // namespace JSC
