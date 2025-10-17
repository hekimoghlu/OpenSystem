/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

#include "ClassInfo.h"
#include "DOMJITEffect.h"
#include "SpeculatedType.h"
#include <wtf/CodePtr.h>

namespace JSC { namespace DOMJIT {

// FIXME: Currently, we only support functions which arguments are up to 2.
// Eventually, we should extend this. But possibly, 2 or 3 can cover typical use cases.
// https://bugs.webkit.org/show_bug.cgi?id=164346
#define JSC_DOMJIT_SIGNATURE_MAX_ARGUMENTS 2
#define JSC_DOMJIT_SIGNATURE_MAX_ARGUMENTS_INCLUDING_THIS (1 + JSC_DOMJIT_SIGNATURE_MAX_ARGUMENTS)

class Signature {
public:
    using CodePtr = WTF::CodePtr<CFunctionPtrTag>;

    template<typename... Arguments>
    constexpr Signature(CodePtr functionWithoutTypeCheck, const ClassInfo* classInfo, Effect effect, SpeculatedType result, Arguments... arguments)
        : functionWithoutTypeCheckPtr(functionWithoutTypeCheck.untypedFunc())
        , classInfo(classInfo)
        , result(result)
        , arguments {static_cast<SpeculatedType>(arguments)...}
        , argumentCount(sizeof...(Arguments))
        , effect(effect)
    {
    }

    CodePtr functionWithoutTypeCheck() const { return CodePtr(functionWithoutTypeCheckPtr); }

    const WTF_VTBL_FUNCPTR_PTRAUTH(DOMJITSignature) CodePtr::UntypedFunc functionWithoutTypeCheckPtr;
    const ClassInfo* const classInfo;
    const SpeculatedType result;
    const SpeculatedType arguments[JSC_DOMJIT_SIGNATURE_MAX_ARGUMENTS];
    const unsigned argumentCount;
    const Effect effect;
};

} }
