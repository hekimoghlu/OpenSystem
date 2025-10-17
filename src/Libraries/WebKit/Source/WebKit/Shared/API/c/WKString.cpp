/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#include "WKString.h"
#include "WKStringPrivate.h"

#include "WKAPICast.h"
#include <JavaScriptCore/InitializeThreading.h>
#include <JavaScriptCore/OpaqueJSString.h>
#include <WebCore/WebCoreJITOperations.h>
#include <wtf/StdLibExtras.h>
#include <wtf/unicode/UTF8Conversion.h>

WKTypeID WKStringGetTypeID()
{
    return WebKit::toAPI(API::String::APIType);
}

WKStringRef WKStringCreateWithUTF8CString(const char* string)
{
    return WebKit::toAPI(&API::String::create(WTF::String::fromUTF8(string)).leakRef());
}

WKStringRef WKStringCreateWithUTF8CStringWithLength(const char* string, size_t stringLength)
{
    return WebKit::toAPI(&API::String::create(WTF::String::fromUTF8(unsafeMakeSpan(string, stringLength))).leakRef());
}

bool WKStringIsEmpty(WKStringRef stringRef)
{
    return WebKit::toImpl(stringRef)->stringView().isEmpty();
}

size_t WKStringGetLength(WKStringRef stringRef)
{
    return WebKit::toImpl(stringRef)->stringView().length();
}

size_t WKStringGetCharacters(WKStringRef stringRef, WKChar* buffer, size_t bufferLength)
{
    static_assert(sizeof(WKChar) == sizeof(UChar), "Size of WKChar must match size of UChar");

    unsigned unsignedBufferLength = std::min<size_t>(bufferLength, std::numeric_limits<unsigned>::max());
    auto substring = WebKit::toImpl(stringRef)->stringView().left(unsignedBufferLength);

    substring.getCharacters(unsafeMakeSpan(reinterpret_cast<UChar*>(buffer), bufferLength));
    return substring.length();
}

size_t WKStringGetMaximumUTF8CStringSize(WKStringRef stringRef)
{
    return static_cast<size_t>(WebKit::toImpl(stringRef)->stringView().length()) * 3 + 1;
}

enum StrictType { NonStrict = false, Strict = true };

template <StrictType strict>
size_t WKStringGetUTF8CStringImpl(WKStringRef stringRef, char* buffer, size_t bufferSize)
{
    if (!bufferSize)
        return 0;

    auto string = WebKit::toImpl(stringRef)->stringView();

    auto target = unsafeMakeSpan(byteCast<char8_t>(buffer), bufferSize);
    WTF::Unicode::ConversionResult<char8_t> result;
    if (string.is8Bit())
        result = WTF::Unicode::convert(string.span8(), target.first(bufferSize - 1));
    else {
        if constexpr (strict == NonStrict)
            result = WTF::Unicode::convertReplacingInvalidSequences(string.span16(), target.first(bufferSize - 1));
        else {
            result = WTF::Unicode::convert(string.span16(), target.first(bufferSize - 1));
            if (result.code == WTF::Unicode::ConversionResultCode::SourceInvalid)
                return 0;
        }
    }

    target[result.buffer.size()] = '\0';
    return result.buffer.size() + 1;
}

size_t WKStringGetUTF8CString(WKStringRef stringRef, char* buffer, size_t bufferSize)
{
    return WKStringGetUTF8CStringImpl<StrictType::Strict>(stringRef, buffer, bufferSize);
}

size_t WKStringGetUTF8CStringNonStrict(WKStringRef stringRef, char* buffer, size_t bufferSize)
{
    return WKStringGetUTF8CStringImpl<StrictType::NonStrict>(stringRef, buffer, bufferSize);
}

bool WKStringIsEqual(WKStringRef aRef, WKStringRef bRef)
{
    return WebKit::toImpl(aRef)->stringView() == WebKit::toImpl(bRef)->stringView();
}

bool WKStringIsEqualToUTF8CString(WKStringRef aRef, const char* b)
{
    // FIXME: Should we add a fast path that avoids memory allocation when the string is all ASCII?
    // FIXME: We can do even the general case more efficiently if we write a function in StringView that understands UTF-8 C strings.
    return WebKit::toImpl(aRef)->stringView() == WTF::String::fromUTF8(b);
}

bool WKStringIsEqualToUTF8CStringIgnoringCase(WKStringRef aRef, const char* b)
{
    // FIXME: Should we add a fast path that avoids memory allocation when the string is all ASCII?
    // FIXME: We can do even the general case more efficiently if we write a function in StringView that understands UTF-8 C strings.
    return equalIgnoringASCIICase(WebKit::toImpl(aRef)->stringView(), WTF::String::fromUTF8(b));
}

WKStringRef WKStringCreateWithJSString(JSStringRef jsStringRef)
{
    auto apiString = jsStringRef ? API::String::create(jsStringRef->string()) : API::String::createNull();

    return WebKit::toAPI(&apiString.leakRef());
}

JSStringRef WKStringCopyJSString(WKStringRef stringRef)
{
    JSC::initialize();
    WebCore::populateJITOperations();
    return OpaqueJSString::tryCreate(WebKit::toImpl(stringRef)->string()).leakRef();
}
