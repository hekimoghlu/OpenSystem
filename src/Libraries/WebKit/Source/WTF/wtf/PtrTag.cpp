/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include <wtf/PtrTag.h>

#include <wtf/WTFConfig.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

#if CPU(ARM64E) && ENABLE(PTRTAG_DEBUGGING)

const char* tagForPtr(const void* ptr)
{
    PtrTagLookup* lookup = g_wtfConfig.ptrTagLookupHead;
    while (lookup) {
        const char* tagName = lookup->tagForPtr(ptr);
        if (tagName)
            return tagName;
        lookup = lookup->next;
    }

    if (ptr == removeCodePtrTag(ptr))
        return "NoPtrTag";

#define RETURN_NAME_IF_TAG_MATCHES(tag) \
    if (ptr == tagCodePtrImpl<PtrTagAction::NoAssert, tag>(removeCodePtrTag(ptr))) \
        return #tag;
    FOR_EACH_WTF_PTRTAG(RETURN_NAME_IF_TAG_MATCHES)
#undef RETURN_NAME_IF_TAG_MATCHES

    return "<unknown PtrTag>";
}

const char* ptrTagName(PtrTag tag)
{
    PtrTagLookup* lookup = g_wtfConfig.ptrTagLookupHead;
    while (lookup) {
        const char* tagName = lookup->ptrTagName(tag);
        if (tagName)
            return tagName;
        lookup = lookup->next;
    }

#define RETURN_WTF_PTRTAG_NAME(_tagName) case _tagName: return #_tagName;
    switch (tag) {
        FOR_EACH_WTF_PTRTAG(RETURN_WTF_PTRTAG_NAME)
    default: return "<unknown>";
    }
#undef RETURN_WTF_PTRTAG_NAME
}

void registerPtrTagLookup(PtrTagLookup* lookup)
{
    lookup->next = g_wtfConfig.ptrTagLookupHead;
    g_wtfConfig.ptrTagLookupHead = lookup;
}

void reportBadTag(const void* ptr, PtrTag expectedTag)
{
    dataLog("PtrTag ASSERTION FAILED on pointer ", RawPointer(ptr), ", actual tag = ", tagForPtr(ptr));
    if (expectedTag == AnyPtrTag)
        dataLogLn(", expected any tag but NoPtrTag");
    else
        dataLogLn(", expected tag = ", ptrTagName(expectedTag));
}

#endif // CPU(ARM64E) && ENABLE(PTRTAG_DEBUGGING)

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
