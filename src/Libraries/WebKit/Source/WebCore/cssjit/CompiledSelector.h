/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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

#if ENABLE(CSS_SELECTOR_JIT)

#include "CSSSelector.h"
#include <JavaScriptCore/JSCPtrTag.h>
#include <JavaScriptCore/MacroAssemblerCodeRef.h>

#define CSS_SELECTOR_JIT_PROFILING 0

namespace WebCore {

enum class SelectorCompilationStatus : uint8_t {
    NotCompiled,
    CannotCompile,
    SimpleSelectorChecker,
    SelectorCheckerWithCheckingContext
};

struct CompiledSelector {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    SelectorCompilationStatus status { SelectorCompilationStatus::NotCompiled };
    JSC::MacroAssemblerCodeRef<JSC::CSSSelectorPtrTag> codeRef;

#if defined(CSS_SELECTOR_JIT_PROFILING) && CSS_SELECTOR_JIT_PROFILING
    unsigned useCount { 0 };
    const CSSSelector* selector { nullptr };
    void wasUsed() { ++useCount; }

    ~CompiledSelector()
    {
        if (codeRef.code().taggedPtr())
            dataLogF("CompiledSelector %d \"%s\"\n", useCount, selector->selectorText().utf8().data());
    }
#else
    void wasUsed() { }
#endif
};

}

#endif
