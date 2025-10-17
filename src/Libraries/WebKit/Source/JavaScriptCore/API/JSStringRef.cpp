/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#include "JSStringRef.h"
#include "JSStringRefPrivate.h"

#include "InitializeThreading.h"
#include "OpaqueJSString.h"
#include <wtf/unicode/UTF8Conversion.h>

using namespace JSC;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

JSStringRef JSStringCreateWithCharacters(const JSChar* chars, size_t numChars)
{
    JSC::initialize();
    return &OpaqueJSString::create({ reinterpret_cast<const UChar*>(chars), numChars }).leakRef();
}

JSStringRef JSStringCreateWithUTF8CString(const char* string)
{
    JSC::initialize();
    if (string) {
        auto stringSpan = unsafeSpan8(string);
        Vector<UChar, 1024> buffer(stringSpan.size());
        auto result = WTF::Unicode::convert(spanReinterpretCast<const char8_t>(stringSpan), buffer.mutableSpan());
        if (result.code == WTF::Unicode::ConversionResultCode::Success) {
            if (result.isAllASCII)
                return &OpaqueJSString::create(stringSpan).leakRef();
            return &OpaqueJSString::create(result.buffer).leakRef();
        }
    }

    return &OpaqueJSString::create().leakRef();
}

JSStringRef JSStringCreateWithCharactersNoCopy(const JSChar* chars, size_t numChars)
{
    JSC::initialize();
    return OpaqueJSString::tryCreate(StringImpl::createWithoutCopying({ reinterpret_cast<const UChar*>(chars), numChars })).leakRef();
}

JSStringRef JSStringRetain(JSStringRef string)
{
    string->ref();
    return string;
}

void JSStringRelease(JSStringRef string)
{
    string->deref();
}

size_t JSStringGetLength(JSStringRef string)
{
    if (!string)
        return 0;
    return string->length();
}

const JSChar* JSStringGetCharactersPtr(JSStringRef string)
{
    if (!string)
        return nullptr;
    return reinterpret_cast<const JSChar*>(string->characters());
}

size_t JSStringGetMaximumUTF8CStringSize(JSStringRef string)
{
    // Any UTF8 character > 3 bytes encodes as a UTF16 surrogate pair.
    return string->length() * 3 + 1; // + 1 for terminating '\0'
}

size_t JSStringGetUTF8CString(JSStringRef string, char* buffer, size_t bufferSize)
{
    if (!string || !buffer || !bufferSize)
        return 0;

    std::span<char8_t> target { byteCast<char8_t>(buffer), bufferSize - 1 };
    WTF::Unicode::ConversionResult<char8_t> result;
    if (string->is8Bit())
        result = WTF::Unicode::convert(string->span8(), target);
    else
        result = WTF::Unicode::convert(string->span16(), target);
    buffer[result.buffer.size()] = '\0';
    return result.buffer.size() + 1;
}

bool JSStringIsEqual(JSStringRef a, JSStringRef b)
{
    return OpaqueJSString::equal(a, b);
}

bool JSStringIsEqualToUTF8CString(JSStringRef a, const char* b)
{
    return JSStringIsEqual(a, adoptRef(JSStringCreateWithUTF8CString(b)).get());
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
