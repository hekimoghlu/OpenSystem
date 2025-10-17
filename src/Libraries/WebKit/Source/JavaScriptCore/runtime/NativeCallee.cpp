/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#include "config.h"
#include "NativeCallee.h"

#include "LLIntExceptions.h"
#include "NativeCalleeRegistry.h"
#include "WasmCallee.h"

namespace JSC {

NativeCallee::NativeCallee(Category category, ImplementationVisibility implementationVisibility)
    : m_category(category)
    , m_implementationVisibility(implementationVisibility)
{
}

void NativeCallee::dump(PrintStream& out) const
{
    switch (category()) {
    case Category::Wasm: {
#if ENABLE(WEBASSEMBLY)
        static_cast<const Wasm::Callee*>(this)->dump(out);
#endif
        break;
    }
    case Category::InlineCache: {
        out.print(RawPointer(this));
        break;
    }
    }
}

void NativeCallee::operator delete(NativeCallee* callee, std::destroying_delete_t)
{
    NativeCalleeRegistry::singleton().unregisterCallee(callee);
    switch (callee->category()) {
    case Category::Wasm: {
#if ENABLE(WEBASSEMBLY)
        Wasm::Callee::destroy(static_cast<Wasm::Callee*>(callee));
#endif
        break;
    }
    case Category::InlineCache: {
        break;
    }
    }
}

} // namespace JSC
