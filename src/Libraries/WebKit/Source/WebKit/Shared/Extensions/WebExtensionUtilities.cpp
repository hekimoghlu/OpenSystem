/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#include "WebExtensionUtilities.h"
#include <wtf/text/MakeString.h>

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

Ref<JSON::Array> filterObjects(const JSON::Array& array, WTF::Function<bool(const JSON::Value&)>&& lambda)
{
    auto result = JSON::Array::create();

    for (Ref value : array) {
        if (!value)
            continue;

        if (lambda(value))
            result->pushValue(WTFMove(value));
    }

    return result;
}

Vector<String> makeStringVector(const JSON::Array& array)
{
    Vector<String> vector;
    size_t count = array.length();
    vector.reserveInitialCapacity(count);

    for (Ref value : array) {
        if (auto string = value->asString(); !string.isNull())
            vector.append(WTFMove(string));
    }

    vector.shrinkToFit();
    return vector;
}

double largestDisplayScale()
{
    auto largestDisplayScale = 1.0;

    for (double scale : availableScreenScales()) {
        if (scale > largestDisplayScale)
            largestDisplayScale = scale;
    }

    return largestDisplayScale;
}

RefPtr<JSON::Object> jsonWithLowercaseKeys(RefPtr<JSON::Object> json)
{
    if (!json)
        return json;

    Ref newObject = JSON::Object::create();
    for (auto& key : json->keys())
        newObject->setValue(key.convertToASCIILowercase(), *json->getValue(key));

    return newObject;
}

RefPtr<JSON::Object> mergeJSON(RefPtr<JSON::Object> jsonA, RefPtr<JSON::Object> jsonB)
{
    if (!jsonA || !jsonB)
        return jsonA ?: jsonB;

    RefPtr mergedObject = jsonA.copyRef();
    for (auto& key : jsonB->keys()) {
        if (!mergedObject->getValue(key))
            mergedObject->setValue(key, *jsonB->getValue(key));
    }

    return mergedObject;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

WTF_ATTRIBUTE_PRINTF(1, 0)
static String formatString(const char* format, va_list arguments)
{
    va_list args;
    va_copy(args, arguments);

ALLOW_NONLITERAL_FORMAT_BEGIN

#if PLATFORM(COCOA)
    auto cfFormat = adoptCF(CFStringCreateWithCStringNoCopy(kCFAllocatorDefault, format, kCFStringEncodingUTF8, kCFAllocatorNull));
    auto cfResult = adoptCF(CFStringCreateWithFormatAndArguments(0, 0, cfFormat.get(), args));
    va_end(args);
    return cfResult.get();
#endif

#if PLATFORM(WIN)
    int len = _vscwprintf(format, args);
    Vector<wchar_t> buffer(len + 1);
    _vsnwprintf(buffer.data(), len + 1, format, args);
    va_end(args);
    return { buffer.data() };
#else
    char ch;
    int result = vsnprintf(&ch, 1, format, args);

    if (!result) {
        va_end(args);
        return emptyString();
    }

    if (result < 0) {
        va_end(args);
        return nullString();
    }

    Vector<char, 256> buffer;
    buffer.grow(result + 1);

    vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);

    return StringImpl::create(buffer.subspan(0, buffer.size() - 1));
#endif

ALLOW_NONLITERAL_FORMAT_END
}

WTF_ATTRIBUTE_PRINTF(1, 0)
static String formatString(const char* format, ...)
{
    va_list args;
    va_start(args, format);
ALLOW_NONLITERAL_FORMAT_BEGIN
    auto result = formatString(format, args);
ALLOW_NONLITERAL_FORMAT_END
    va_end(args);
    return result;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

static inline String lowercaseFirst(const String& input)
{
    return !input.isEmpty() ? makeString(input.left(1).convertToASCIILowercase(), input.substring(1, input.length())) : input;
}

static inline String uppercaseFirst(const String& input)
{
    return !input.isEmpty() ? makeString(input.left(1).convertToASCIIUppercase(), input.substring(1, input.length())) : input;
}

String toErrorString(const String& callingAPIName, const String& sourceKey, String underlyingErrorString, ...)
{
    ASSERT(!underlyingErrorString.isEmpty());

    va_list arguments;
    va_start(arguments, underlyingErrorString);

ALLOW_NONLITERAL_FORMAT_BEGIN
    String formattedUnderlyingErrorString = formatString(underlyingErrorString.utf8().data(), arguments).trim([](UChar character) -> bool {
        return character == '.';
    });
ALLOW_NONLITERAL_FORMAT_END

    va_end(arguments);

    String source = sourceKey;

    if (UNLIKELY(!callingAPIName.isEmpty() && !sourceKey.isEmpty() && formattedUnderlyingErrorString.contains("value is invalid"_s))) {
        ASSERT_NOT_REACHED_WITH_MESSAGE("Overly nested error string, use a `nullString()` sourceKey for this call instead.");
        source = nullString();
    }

    if (!callingAPIName.isEmpty() && !source.isEmpty())
        return formatString("Invalid call to %s. The '%s' value is invalid, because %s.", callingAPIName.utf8().data(), source.utf8().data(), lowercaseFirst(formattedUnderlyingErrorString).utf8().data());

    if (callingAPIName.isEmpty() && !source.isEmpty())
        return formatString("The '%s' value is invalid, because %s.", source.utf8().data(), lowercaseFirst(formattedUnderlyingErrorString).utf8().data());

    if (!callingAPIName.isEmpty())
        return formatString("Invalid call to %s. %s.", callingAPIName.utf8().data(), uppercaseFirst(formattedUnderlyingErrorString).utf8().data());

    return formattedUnderlyingErrorString;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
