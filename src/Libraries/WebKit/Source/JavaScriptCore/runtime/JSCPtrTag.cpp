/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "JSCPtrTag.h"

#include "JSCConfig.h"

#if ENABLE(JIT_CAGE)
#include <machine/cpu_capabilities.h>
#include <sys/mman.h>
#include <sys/types.h>
#endif

namespace JSC {

#if CPU(ARM64E) && (ENABLE(PTRTAG_DEBUGGING) || ENABLE(DISASSEMBLER))

const char* ptrTagName(PtrTag tag)
{
#define RETURN_PTRTAG_NAME(_tagName, calleeType, callerType) case _tagName: return #_tagName;
    switch (static_cast<unsigned>(tag)) {
        FOR_EACH_JSC_PTRTAG(RETURN_PTRTAG_NAME)
    }
#undef RETURN_PTRTAG_NAME
    return nullptr; // Matching tag not found.
}

#if ENABLE(PTRTAG_DEBUGGING)
static const char* tagForPtr(const void* ptr)
{
#define RETURN_NAME_IF_TAG_MATCHES(tag, calleeType, callerType) \
    if (callerType != PtrTagCallerType::JIT || calleeType != PtrTagCalleeType::Native) { \
        if (ptr == WTF::tagCodePtrImpl<WTF::PtrTagAction::NoAssert, JSC::tag>(removeCodePtrTag(ptr))) \
            return #tag; \
    }
    FOR_EACH_JSC_PTRTAG(RETURN_NAME_IF_TAG_MATCHES)
#undef RETURN_NAME_IF_TAG_MATCHES
    return nullptr; // Matching tag not found.
}

void initializePtrTagLookup()
{
    WTF::PtrTagLookup& lookup = g_jscConfig.ptrTagLookupRecord;
    lookup.initialize(tagForPtr, ptrTagName);
    WTF::registerPtrTagLookup(&lookup);
}
#endif // ENABLE(PTRTAG_DEBUGGING)
#endif // CPU(ARM64E) && (ENABLE(PTRTAG_DEBUGGING) || ENABLE(DISASSEMBLER))

#if CPU(ARM64E)

PtrTagCallerType callerType(PtrTag tag)
{
#define RETURN_PTRTAG_TYPE(_tagName, calleeType, callerType) case _tagName: return callerType;
    switch (tag) {
        FOR_EACH_JSC_PTRTAG(RETURN_PTRTAG_TYPE)
    default:
        return PtrTagCallerType::Native;
    }
#undef RETURN_PTRTAG_TYPE
    return PtrTagCallerType::Native;
}

PtrTagCalleeType calleeType(PtrTag tag)
{
#define RETURN_PTRTAG_TYPE(_tagName, calleeType, callerType) case _tagName: return calleeType;
    switch (tag) {
        FOR_EACH_JSC_PTRTAG(RETURN_PTRTAG_TYPE)
    default:
        return PtrTagCalleeType::Native;
    }
#undef RETURN_PTRTAG_TYPE
    return PtrTagCalleeType::Native;
}

#endif

} // namespace JSC
