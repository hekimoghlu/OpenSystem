/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

#include "JSFunction.h"

namespace JSC {

class JSPromise;
class JSPromisePrototype;
class GetterSetter;

class JSPromiseConstructor : public JSFunction {
public:
    using Base = JSFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static JSPromiseConstructor* create(VM&, Structure*, JSPromisePrototype*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

protected:
    JSPromiseConstructor(VM&, FunctionExecutable*, JSGlobalObject*, Structure*);
    void finishCreation(VM&, JSPromisePrototype*);

private:
    void addOwnInternalSlots(VM&, JSGlobalObject*);
};
static_assert(sizeof(JSPromiseConstructor) == sizeof(JSFunction), "Allocate JSPromiseConstructor in JSFunction IsoSubspace");

} // namespace JSC
