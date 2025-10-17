/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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

#if ENABLE(WEBASSEMBLY)

#include "InternalFunction.h"
#include "JSObject.h"

#include <wtf/Vector.h>

namespace JSC {

class JSWebAssemblyModule;
class WebAssemblyModulePrototype;

class WebAssemblyModuleConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static WebAssemblyModuleConstructor* create(VM&, Structure*, WebAssemblyModulePrototype*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static JSWebAssemblyModule* createModule(JSGlobalObject*, CallFrame*, Vector<uint8_t>&& buffer);

private:
    WebAssemblyModuleConstructor(VM&, Structure*);
    void finishCreation(VM&, WebAssemblyModulePrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(WebAssemblyModuleConstructor, InternalFunction);

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
