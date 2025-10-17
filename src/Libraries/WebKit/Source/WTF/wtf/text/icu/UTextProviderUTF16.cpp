/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#include <wtf/text/icu/UTextProviderUTF16.h>

#include <algorithm>
#include <wtf/text/icu/UTextProvider.h>

namespace WTF {

// UTF16ContextAware provider

static UText* uTextUTF16ContextAwareClone(UText*, const UText*, UBool, UErrorCode*);
static int64_t uTextUTF16ContextAwareNativeLength(UText*);
static UBool uTextUTF16ContextAwareAccess(UText*, int64_t, UBool);
static int32_t uTextUTF16ContextAwareExtract(UText*, int64_t, int64_t, UChar*, int32_t, UErrorCode*);
static void uTextUTF16ContextAwareClose(UText*);

static const struct UTextFuncs textUTF16ContextAwareFuncs = {
    sizeof(UTextFuncs),
    0,
    0,
    0,
    uTextUTF16ContextAwareClone,
    uTextUTF16ContextAwareNativeLength,
    uTextUTF16ContextAwareAccess,
    uTextUTF16ContextAwareExtract,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    uTextUTF16ContextAwareClose,
    nullptr,
    nullptr,
    nullptr
};

static inline UTextProviderContext textUTF16ContextAwareGetCurrentContext(const UText* text)
{
    if (!text->chunkContents)
        return UTextProviderContext::NoContext;
    return text->chunkContents == text->p ? UTextProviderContext::PrimaryContext : UTextProviderContext::PriorContext;
}

static void textUTF16ContextAwareMoveInPrimaryContext(UText* text, int64_t nativeIndex, int64_t nativeLength, UBool forward)
{
    ASSERT(text->chunkContents == text->p);
    ASSERT_UNUSED(forward, forward ? nativeIndex >= text->b : nativeIndex > text->b);
    ASSERT(nativeIndex <= nativeLength);
    text->chunkNativeStart = text->b;
    text->chunkNativeLimit = nativeLength;
    int64_t length = text->chunkNativeLimit - text->chunkNativeStart;
    // Ensure chunk length is well defined if computed length exceeds int32_t range.
    ASSERT(length < std::numeric_limits<int32_t>::max());
    text->chunkLength = length < std::numeric_limits<int32_t>::max() ? static_cast<int32_t>(length) : 0;
    text->nativeIndexingLimit = text->chunkLength;
    int64_t offset = nativeIndex - text->chunkNativeStart;
    // Ensure chunk offset is well defined if computed offset exceeds int32_t range or chunk length.
    ASSERT(offset < std::numeric_limits<int32_t>::max());
    text->chunkOffset = std::min(offset < std::numeric_limits<int32_t>::max() ? static_cast<int32_t>(offset) : 0, text->chunkLength);
}

static void textUTF16ContextAwareSwitchToPrimaryContext(UText* text, int64_t nativeIndex, int64_t nativeLength, UBool forward)
{
    ASSERT(!text->chunkContents || text->chunkContents == text->q);
    text->chunkContents = static_cast<const UChar*>(text->p);
    textUTF16ContextAwareMoveInPrimaryContext(text, nativeIndex, nativeLength, forward);
}

static void textUTF16ContextAwareMoveInPriorContext(UText* text, int64_t nativeIndex, int64_t nativeLength, UBool forward)
{
    ASSERT(text->chunkContents == text->q);
    ASSERT_UNUSED(forward, forward ? nativeIndex < text->b : nativeIndex <= text->b);
    ASSERT_UNUSED(nativeLength, nativeIndex <= nativeLength);
    text->chunkNativeStart = 0;
    text->chunkNativeLimit = text->b;
    text->chunkLength = text->b;
    text->nativeIndexingLimit = text->chunkLength;
    int64_t offset = nativeIndex - text->chunkNativeStart;
    // Ensure chunk offset is well defined if computed offset exceeds int32_t range or chunk length.
    ASSERT(offset < std::numeric_limits<int32_t>::max());
    text->chunkOffset = std::min(offset < std::numeric_limits<int32_t>::max() ? static_cast<int32_t>(offset) : 0, text->chunkLength);
}

static void textUTF16ContextAwareSwitchToPriorContext(UText* text, int64_t nativeIndex, int64_t nativeLength, UBool forward)
{
    ASSERT(!text->chunkContents || text->chunkContents == text->p);
    text->chunkContents = static_cast<const UChar*>(text->q);
    textUTF16ContextAwareMoveInPriorContext(text, nativeIndex, nativeLength, forward);
}

static UText* uTextUTF16ContextAwareClone(UText* destination, const UText* source, UBool deep, UErrorCode* status)
{
    return uTextCloneImpl(destination, source, deep, status);
}

static inline int64_t uTextUTF16ContextAwareNativeLength(UText* text)
{
    return text->a + text->b;
}

static UBool uTextUTF16ContextAwareAccess(UText* text, int64_t nativeIndex, UBool forward)
{
    if (!text->context)
        return false;
    int64_t nativeLength = uTextUTF16ContextAwareNativeLength(text);
    UBool isAccessible;
    if (uTextAccessInChunkOrOutOfRange(text, nativeIndex, nativeLength, forward, isAccessible))
        return isAccessible;
    nativeIndex = uTextAccessPinIndex(nativeIndex, nativeLength);
    UTextProviderContext currentContext = textUTF16ContextAwareGetCurrentContext(text);
    UTextProviderContext newContext = uTextProviderContext(text, nativeIndex, forward);
    ASSERT(newContext != UTextProviderContext::NoContext);
    if (newContext == currentContext) {
        if (currentContext == UTextProviderContext::PrimaryContext)
            textUTF16ContextAwareMoveInPrimaryContext(text, nativeIndex, nativeLength, forward);
        else
            textUTF16ContextAwareMoveInPriorContext(text, nativeIndex, nativeLength, forward);
    } else if (newContext == UTextProviderContext::PrimaryContext)
        textUTF16ContextAwareSwitchToPrimaryContext(text, nativeIndex, nativeLength, forward);
    else {
        ASSERT(newContext == UTextProviderContext::PriorContext);
        textUTF16ContextAwareSwitchToPriorContext(text, nativeIndex, nativeLength, forward);
    }
    return true;
}

static int32_t uTextUTF16ContextAwareExtract(UText*, int64_t, int64_t, UChar*, int32_t, UErrorCode* errorCode)
{
    // In the present context, this text provider is used only with ICU functions
    // that do not perform an extract operation.
    ASSERT_NOT_REACHED();
    *errorCode = U_UNSUPPORTED_ERROR;
    return 0;
}

static void uTextUTF16ContextAwareClose(UText* text)
{
    text->context = nullptr;
}

UText* openUTF16ContextAwareUTextProvider(UText* text, std::span<const UChar> string, std::span<const UChar> priorContext, UErrorCode* status)
{
    if (U_FAILURE(*status))
        return nullptr;
    if (!string.data() || string.size() > static_cast<unsigned>(std::numeric_limits<int32_t>::max())) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return nullptr;
    }
    text = utext_setup(text, 0, status);
    if (U_FAILURE(*status)) {
        ASSERT(!text);
        return nullptr;
    }

    initializeContextAwareUTextProvider(text, &textUTF16ContextAwareFuncs, string, priorContext);
    return text;
}

} // namespace WTF
