/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include "WasmFormat.h"
#include "WasmOps.h"

namespace JSC {

class JSWebAssemblyTag;
class WebAssemblyTagPrototype;

namespace Wasm {
class Tag;
}

class WebAssemblyTagConstructor final : public InternalFunction {
public:
    using Base = InternalFunction;

    static WebAssemblyTagConstructor* create(VM&, Structure*, WebAssemblyTagPrototype*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    WebAssemblyTagConstructor(VM&, Structure*);
    void finishCreation(VM&, WebAssemblyTagPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(WebAssemblyTagConstructor, InternalFunction);

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
