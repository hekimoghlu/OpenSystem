/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

#include "DOMJITCallDOMGetterSnippet.h"
#include "PropertySlot.h"
#include "PutPropertySlot.h"
#include "SpeculatedType.h"

namespace JSC { namespace DOMJIT {

class CallDOMGetterSnippet;

class GetterSetter {
public:
    using CustomGetter = GetValueFunc::Ptr;
    using CustomSetter = PutValueFunc::Ptr;
    using SnippetCompiler = Ref<DOMJIT::CallDOMGetterSnippet>(*)();

    constexpr GetterSetter(CustomGetter getter, SnippetCompiler compiler, SpeculatedType resultType)
        : m_getter(getter)
        , m_compiler(compiler)
        , m_resultType(resultType)
    {
    }

    constexpr CustomGetter getter() const { return m_getter; }
    constexpr CustomSetter setter() const { return nullptr; }
    constexpr SnippetCompiler compiler() const { return m_compiler; }
    constexpr SpeculatedType resultType() const { return m_resultType; }

private:
    WTF_VTBL_FUNCPTR_PTRAUTH(DOMJITGetterSetter) CustomGetter m_getter;
    SnippetCompiler m_compiler;
    SpeculatedType m_resultType;
};

} }
