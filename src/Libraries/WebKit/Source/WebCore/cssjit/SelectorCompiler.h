/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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

#include "CompiledSelector.h"
#include "SelectorChecker.h"
#include <JavaScriptCore/LLIntThunks.h>
#include <JavaScriptCore/VM.h>

namespace WebCore {

class CSSSelector;
class Element;

namespace SelectorCompiler {

enum class SelectorContext {
    // Rule Collector needs a resolvingMode and can modify the tree as it matches.
    RuleCollector,

    // Query Selector does not modify the tree and never match :visited.
    QuerySelector
};

void compileSelector(CompiledSelector&, const CSSSelector*, SelectorContext);

inline unsigned ruleCollectorSimpleSelectorChecker(CompiledSelector& compiledSelector, const Element* element, unsigned* value)
{
    ASSERT(compiledSelector.status == SelectorCompilationStatus::SimpleSelectorChecker);
#if CPU(ARM64E) && !ENABLE(C_LOOP)
    if (JSC::Options::useJITCage())
        return JSC::vmEntryToCSSJIT(std::bit_cast<uintptr_t>(element), std::bit_cast<uintptr_t>(value), 0, compiledSelector.codeRef.code().taggedPtr());
#endif
    using RuleCollectorSimpleSelectorChecker = unsigned SYSV_ABI (*)(const Element*, unsigned*);
    return untagCFunctionPtr<RuleCollectorSimpleSelectorChecker, JSC::CSSSelectorPtrTag>(compiledSelector.codeRef.code().taggedPtr())(element, value);
}

inline unsigned querySelectorSimpleSelectorChecker(CompiledSelector& compiledSelector, const Element* element)
{
    ASSERT(compiledSelector.status == SelectorCompilationStatus::SimpleSelectorChecker);
#if CPU(ARM64E) && !ENABLE(C_LOOP)
    if (JSC::Options::useJITCage())
        return JSC::vmEntryToCSSJIT(std::bit_cast<uintptr_t>(element), 0, 0, compiledSelector.codeRef.code().taggedPtr());
#endif
    using QuerySelectorSimpleSelectorChecker = unsigned SYSV_ABI (*)(const Element*);
    return untagCFunctionPtr<QuerySelectorSimpleSelectorChecker, JSC::CSSSelectorPtrTag>(compiledSelector.codeRef.code().taggedPtr())(element);
}

inline unsigned ruleCollectorSelectorCheckerWithCheckingContext(CompiledSelector& compiledSelector, const Element* element, SelectorChecker::CheckingContext* context, unsigned* value)
{
    ASSERT(compiledSelector.status == SelectorCompilationStatus::SelectorCheckerWithCheckingContext);
#if CPU(ARM64E) && !ENABLE(C_LOOP)
    if (JSC::Options::useJITCage())
        return JSC::vmEntryToCSSJIT(std::bit_cast<uintptr_t>(element), std::bit_cast<uintptr_t>(context), std::bit_cast<uintptr_t>(value), compiledSelector.codeRef.code().taggedPtr());
#endif
    using RuleCollectorSelectorCheckerWithCheckingContext = unsigned SYSV_ABI (*)(const Element*, SelectorChecker::CheckingContext*, unsigned*);
    return untagCFunctionPtr<RuleCollectorSelectorCheckerWithCheckingContext, JSC::CSSSelectorPtrTag>(compiledSelector.codeRef.code().taggedPtr())(element, context, value);
}

inline unsigned querySelectorSelectorCheckerWithCheckingContext(CompiledSelector& compiledSelector, const Element* element, const SelectorChecker::CheckingContext* context)
{
    ASSERT(compiledSelector.status == SelectorCompilationStatus::SelectorCheckerWithCheckingContext);
#if CPU(ARM64E) && !ENABLE(C_LOOP)
    if (JSC::Options::useJITCage())
        return JSC::vmEntryToCSSJIT(std::bit_cast<uintptr_t>(element), std::bit_cast<uintptr_t>(context), 0, compiledSelector.codeRef.code().taggedPtr());
#endif
    using QuerySelectorSelectorCheckerWithCheckingContext = unsigned SYSV_ABI (*)(const Element*, const SelectorChecker::CheckingContext*);
    return untagCFunctionPtr<QuerySelectorSelectorCheckerWithCheckingContext, JSC::CSSSelectorPtrTag>(compiledSelector.codeRef.code().taggedPtr())(element, context);
}

} // namespace SelectorCompiler
} // namespace WebCore

#endif // ENABLE(CSS_SELECTOR_JIT)
