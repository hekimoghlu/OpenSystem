/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

#include "Options.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/PtrTag.h>

namespace JSC {

#if ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

// This indirection is provided so that we can manually force on assertions for
// testing even on release builds.
#if ENABLE(JIT_OPERATION_VALIDATION) && ASSERT_ENABLED
#define ENABLE_JIT_OPERATION_VALIDATION_ASSERT 1
#endif

struct JITOperationAnnotation;

class JITOperationList {
public:
    static JITOperationList& singleton();
    static void initialize();

#if ENABLE(JIT_OPERATION_VALIDATION)
    template<typename PtrType>
    void* map(PtrType pointer) const
    {
        return m_validatedOperations.get(removeCodePtrTag(std::bit_cast<void*>(pointer)));
    }

#if ENABLE(JIT_OPERATION_VALIDATION_ASSERT)
    template<typename PtrType>
    void* inverseMap(PtrType pointer) const
    {
        return m_validatedOperationsInverseMap.get(std::bit_cast<void*>(pointer));
    }
#endif

    JS_EXPORT_PRIVATE static void populatePointersInEmbedder(const JITOperationAnnotation* beginOperations, const JITOperationAnnotation* endOperations);
#endif // ENABLE(JIT_OPERATION_VALIDATION)

    static void populatePointersInJavaScriptCore();
    static void populatePointersInJavaScriptCoreForLLInt();

#if ENABLE(JIT_OPERATION_DISASSEMBLY)
    JS_EXPORT_PRIVATE static void populateDisassemblyLabelsInEmbedder(const JITOperationAnnotation* beginOperations, const JITOperationAnnotation* endOperations);
#endif

    template<typename T> static void assertIsJITOperation(T function)
    {
        UNUSED_PARAM(function);
#if ENABLE(JIT_OPERATION_VALIDATION_ASSERT)
        RELEASE_ASSERT(!Options::useJIT() || JITOperationList::singleton().map(function));
#endif
    }

    template<typename T> static void assertIsJITOperationWithValidation(T function)
    {
        UNUSED_PARAM(function);
#if ENABLE(JIT_OPERATION_VALIDATION_ASSERT)
        RELEASE_ASSERT(!Options::useJIT() || JITOperationList::singleton().inverseMap(function));
#endif
    }

private:
#if ENABLE(JIT_OPERATION_DISASSEMBLY)
    static void populateDisassemblyLabelsInJavaScriptCore();
    static void populateDisassemblyLabelsInJavaScriptCoreForLLInt();
    static void addDisassemblyLabels(const JITOperationAnnotation* begin, const JITOperationAnnotation* end);
#endif

#if ENABLE(JIT_OPERATION_VALIDATION)
    ALWAYS_INLINE void addPointers(const JITOperationAnnotation* begin, const JITOperationAnnotation* end);

#if ENABLE(JIT_OPERATION_VALIDATION_ASSERT)
    void addInverseMap(void* validationEntry, void* pointer);
#endif

    UncheckedKeyHashMap<void*, void*> m_validatedOperations;
#if ENABLE(JIT_OPERATION_VALIDATION_ASSERT)
    UncheckedKeyHashMap<void*, void*> m_validatedOperationsInverseMap;
#endif
#endif // ENABLE(JIT_OPERATION_VALIDATION)
};

#if ENABLE(JIT_OPERATION_VALIDATION)

JS_EXPORT_PRIVATE extern LazyNeverDestroyed<JITOperationList> jitOperationList;

inline JITOperationList& JITOperationList::singleton()
{
    return jitOperationList.get();
}

#else // not ENABLE(JIT_OPERATION_VALIDATION)

ALWAYS_INLINE void JITOperationList::populatePointersInJavaScriptCore()
{
    if (UNLIKELY(Options::needDisassemblySupport()))
        populateDisassemblyLabelsInJavaScriptCore();
}

ALWAYS_INLINE void JITOperationList::populatePointersInJavaScriptCoreForLLInt()
{
    if (UNLIKELY(Options::needDisassemblySupport()))
        populateDisassemblyLabelsInJavaScriptCoreForLLInt();
}

#endif // ENABLE(JIT_OPERATION_VALIDATION)

#else // not ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

class JITOperationList {
public:
    static void initialize() { }

    static void populatePointersInJavaScriptCore() { }
    static void populatePointersInJavaScriptCoreForLLInt() { }

    template<typename T> static void assertIsJITOperation(T) { }
    template<typename T> static void assertIsJITOperationWithValidation(T) { }
};

#endif // ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

} // namespace JSC
