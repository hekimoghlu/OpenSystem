/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#include <wtf/text/StringImpl.h>

#if USE(CF)

#include <CoreFoundation/CoreFoundation.h>
#include <wtf/DebugHeap.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RetainPtr.h>
#include <wtf/Threading.h>

namespace WTF {

namespace StringWrapperCFAllocator {

    DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(StringWrapperCFAllocator, WTF_INTERNAL);
    DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StringWrapperCFAllocator);

    static StringImpl* currentString;

    static const void* retain(const void* info)
    {
        return info;
    }

    NO_RETURN_DUE_TO_ASSERT
    static void release(const void*)
    {
        ASSERT_NOT_REACHED();
    }

    static CFStringRef copyDescription(const void*)
    {
        return CFSTR("WTF::String-based allocator");
    }

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    static void* allocate(CFIndex size, CFOptionFlags, void*)
    {
        StringImpl* underlyingString = nullptr;
        if (isMainThread()) {
            underlyingString = currentString;
            if (underlyingString) {
                currentString = nullptr;
                underlyingString->ref(); // Balanced by call to deref in deallocate below.
            }
        }
        StringImpl** header = static_cast<StringImpl**>(StringWrapperCFAllocatorMalloc::malloc(sizeof(StringImpl*) + size));
        *header = underlyingString;
        return header + 1;
    }

    static void* reallocate(void* pointer, CFIndex newSize, CFOptionFlags, void*)
    {
        size_t newAllocationSize = sizeof(StringImpl*) + newSize;
        StringImpl** header = static_cast<StringImpl**>(pointer) - 1;
        ASSERT(!*header);
        header = static_cast<StringImpl**>(StringWrapperCFAllocatorMalloc::realloc(header, newAllocationSize));
        return header + 1;
    }

    static void deallocate(void* pointer, void*)
    {
        StringImpl** header = static_cast<StringImpl**>(pointer) - 1;
        if (!*header)
            StringWrapperCFAllocatorMalloc::free(header);
        else {
            ensureOnMainThread([header] {
                StringImpl* underlyingString = *header;
                ASSERT(underlyingString);
                underlyingString->deref(); // Balanced by call to ref in allocate above.
                StringWrapperCFAllocatorMalloc::free(header);
            });
        }
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    static CFIndex preferredSize(CFIndex size, CFOptionFlags, void*)
    {
        // FIXME: If FastMalloc provided a "good size" callback, we'd want to use it here.
        // Note that this optimization would help performance for strings created with the
        // allocator that are mutable, and those typically are only created by callers who
        // make a new string using the old string's allocator, such as some of the call
        // sites in CFURL.
        return size;
    }

    static CFAllocatorRef allocator()
    {
        ASSERT(isMainThread());
        static NeverDestroyed allocator = [] {
            initializeMainThread();
            CFAllocatorContext context = { 0, nullptr, retain, release, copyDescription, allocate, reallocate, deallocate, preferredSize };
            return adoptCF(CFAllocatorCreate(nullptr, &context));
        }();
        return allocator.get().get();
    }

}

RetainPtr<CFStringRef> StringImpl::createCFString()
{
    if (!m_length || !isMainThread()) {
        if (is8Bit()) {
            auto characters = span8();
            return adoptCF(CFStringCreateWithBytes(nullptr, characters.data(), characters.size(), kCFStringEncodingISOLatin1, false));
        }
        auto characters = span16();
        return adoptCF(CFStringCreateWithCharacters(nullptr, reinterpret_cast<const UniChar*>(characters.data()), characters.size()));
    }
    CFAllocatorRef allocator = StringWrapperCFAllocator::allocator();

    // Put pointer to the StringImpl in a global so the allocator can store it with the CFString.
    ASSERT(!StringWrapperCFAllocator::currentString);
    StringWrapperCFAllocator::currentString = this;

    RetainPtr<CFStringRef> string;
    if (is8Bit()) {
        auto characters = span8();
        string = adoptCF(CFStringCreateWithBytesNoCopy(allocator, characters.data(), characters.size(), kCFStringEncodingISOLatin1, false, kCFAllocatorNull));
    } else {
        auto characters = span16();
        string = adoptCF(CFStringCreateWithCharactersNoCopy(allocator, reinterpret_cast<const UniChar*>(characters.data()), characters.size(), kCFAllocatorNull));
    }
    // CoreFoundation might not have to allocate anything, we clear currentString in case we did not execute allocate().
    StringWrapperCFAllocator::currentString = nullptr;

    return string;
}

// On StringImpl creation we could check if the allocator is the StringWrapperCFAllocator.
// If it is, then we could find the original StringImpl and just return that. But to
// do that we'd have to compute the offset from CFStringRef to the allocated block;
// the CFStringRef is *not* at the start of an allocated block. Testing shows 1000x
// more calls to createCFString than calls to the create functions with the appropriate
// allocator, so it's probably not urgent optimize that case.

}

#endif // USE(CF)
