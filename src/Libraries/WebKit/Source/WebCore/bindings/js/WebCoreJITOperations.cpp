/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#include "WebCoreJITOperations.h"

#include <JavaScriptCore/JITOperationList.h>

namespace WebCore {

#if ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

extern const JSC::JITOperationAnnotation startOfJITOperationsInWebCore __asm("section$start$__DATA_CONST$__jsc_ops");
extern const JSC::JITOperationAnnotation endOfJITOperationsInWebCore __asm("section$end$__DATA_CONST$__jsc_ops");

#if ENABLE(JIT_OPERATION_VALIDATION)
void populateJITOperations()
{
    static std::once_flag onceKey;
    std::call_once(onceKey, [] {
        JSC::JITOperationList::populatePointersInEmbedder(&startOfJITOperationsInWebCore, &endOfJITOperationsInWebCore);
    });
#if ENABLE(JIT_OPERATION_DISASSEMBLY)
    if (UNLIKELY(JSC::Options::needDisassemblySupport()))
        populateDisassemblyLabels();
#endif
}
#endif // ENABLE(JIT_OPERATION_VALIDATION)

#if ENABLE(JIT_OPERATION_DISASSEMBLY)
void populateDisassemblyLabels()
{
    static std::once_flag onceKey;
    std::call_once(onceKey, [] {
        JSC::JITOperationList::populateDisassemblyLabelsInEmbedder(&startOfJITOperationsInWebCore, &endOfJITOperationsInWebCore);
    });
}
#endif // ENABLE(JIT_OPERATION_DISASSEMBLY)

#endif // ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

}
