/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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

#if ENABLE(JIT)

#include "MacroAssemblerCodeRef.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class VM;

class OpaqueByproducts;

// This class is a way to keep the result of a compilation alive and runnable.

class Compilation {
    WTF_MAKE_NONCOPYABLE(Compilation);
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(Compilation, JS_EXPORT_PRIVATE);

public:
    JS_EXPORT_PRIVATE Compilation(MacroAssemblerCodeRef<JITCompilationPtrTag>, std::unique_ptr<OpaqueByproducts>);
    JS_EXPORT_PRIVATE Compilation(Compilation&&);
    JS_EXPORT_PRIVATE ~Compilation();

    CodePtr<JITCompilationPtrTag> code() const { return m_codeRef.code(); }
    MacroAssemblerCodeRef<JITCompilationPtrTag> codeRef() const { return m_codeRef; }
    
    CString disassembly() const { return m_codeRef.disassembly(); }

private:
    MacroAssemblerCodeRef<JITCompilationPtrTag> m_codeRef;
    std::unique_ptr<OpaqueByproducts> m_byproducts;
};

} // namespace JSC

#endif // ENABLE(JIT)
