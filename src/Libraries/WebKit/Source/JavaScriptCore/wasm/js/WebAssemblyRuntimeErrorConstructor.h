/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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

namespace JSC {

class WebAssemblyRuntimeErrorPrototype;

class WebAssemblyRuntimeErrorConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    static WebAssemblyRuntimeErrorConstructor* create(VM&, Structure*, WebAssemblyRuntimeErrorPrototype*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    WebAssemblyRuntimeErrorConstructor(VM&, Structure*);
    void finishCreation(VM&, WebAssemblyRuntimeErrorPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(WebAssemblyRuntimeErrorConstructor, InternalFunction);

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
