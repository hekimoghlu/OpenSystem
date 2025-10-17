/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#include "JSGenericTypedArrayViewConstructor.h"

#include "JSGenericTypedArrayView.h"
#include "JSGlobalObjectInlines.h"
#include "ParseInt.h"
#include <wtf/text/Base64.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

JSC_DEFINE_HOST_FUNCTION(uint8ArrayConstructorFromBase64, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSString* jsString = jsDynamicCast<JSString*>(callFrame->argument(0));
    if (UNLIKELY(!jsString))
        return throwVMTypeError(globalObject, scope, "Uint8Array.fromBase64 requires a string"_s);

    auto alphabet = WTF::Alphabet::Base64;
    auto lastChunkHandling = WTF::LastChunkHandling::Loose;

    JSValue optionsValue = callFrame->argument(1);
    if (!optionsValue.isUndefined()) {
        JSObject* optionsObject = jsDynamicCast<JSObject*>(optionsValue);
        if (UNLIKELY(!optionsValue.isObject()))
            return throwVMTypeError(globalObject, scope, "Uint8Array.fromBase64 requires that options be an object"_s);

        JSValue alphabetValue = optionsObject->get(globalObject, vm.propertyNames->alphabet);
        RETURN_IF_EXCEPTION(scope, { });
        if (!alphabetValue.isUndefined()) {
            JSString* alphabetString = jsDynamicCast<JSString*>(alphabetValue);
            if (UNLIKELY(!alphabetString))
                return throwVMTypeError(globalObject, scope, "Uint8Array.fromBase64 requires that alphabet be \"base64\" or \"base64url\""_s);

            StringView alphabetStringView = alphabetString->view(globalObject);
            RETURN_IF_EXCEPTION(scope, { });
            if (alphabetStringView == "base64url"_s)
                alphabet = WTF::Alphabet::Base64URL;
            else if (alphabetStringView != "base64"_s)
                return throwVMTypeError(globalObject, scope, "Uint8Array.fromBase64 requires that alphabet be \"base64\" or \"base64url\""_s);
        }

        JSValue lastChunkHandlingValue = optionsObject->get(globalObject, vm.propertyNames->lastChunkHandling);
        RETURN_IF_EXCEPTION(scope, { });
        if (!lastChunkHandlingValue.isUndefined()) {
            JSString* lastChunkHandlingString = jsDynamicCast<JSString*>(lastChunkHandlingValue);
            if (UNLIKELY(!lastChunkHandlingString))
                return throwVMTypeError(globalObject, scope, "Uint8Array.fromBase64 requires that lastChunkHandling be \"loose\", \"strict\", or \"stop-before-partial\""_s);

            StringView lastChunkHandlingStringView = lastChunkHandlingString->view(globalObject);
            RETURN_IF_EXCEPTION(scope, { });
            if (lastChunkHandlingStringView == "strict"_s)
                lastChunkHandling = WTF::LastChunkHandling::Strict;
            else if (lastChunkHandlingStringView == "stop-before-partial"_s)
                lastChunkHandling = WTF::LastChunkHandling::StopBeforePartial;
            else if (lastChunkHandlingStringView != "loose"_s)
                return throwVMTypeError(globalObject, scope, "Uint8Array.fromBase64 requires that lastChunkHandling be \"loose\", \"strict\", or \"stop-before-partial\""_s);
        }
    }

    StringView view = jsString->view(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    Vector<uint8_t, 128> output;
    output.grow(maxLengthFromBase64(view));
    auto [shouldThrowError, readLength, writeLength] = fromBase64(view, output.mutableSpan(), alphabet, lastChunkHandling);
    if (UNLIKELY(shouldThrowError == WTF::FromBase64ShouldThrowError::Yes))
        return JSValue::encode(throwSyntaxError(globalObject, scope, "Uint8Array.fromBase64 requires a valid base64 string"_s));

    ASSERT(readLength <= view.length());
    JSUint8Array* uint8Array = JSUint8Array::createUninitialized(globalObject, globalObject->typedArrayStructure(TypeUint8, false), writeLength);
    RETURN_IF_EXCEPTION(scope, { });

    memcpySpan(uint8Array->typedSpan(), output.span().first(writeLength));
    return JSValue::encode(uint8Array);
}

template<typename CharacterType>
inline static WARN_UNUSED_RETURN size_t decodeHexImpl(std::span<CharacterType> span, std::span<uint8_t> result)
{
    ASSERT(span.size() == result.size() * 2);

    // http://0x80.pl/notesen/2022-01-17-validating-hex-parse.html
    auto vectorDecode8 = [&](simde_uint8x16_t input, uint8_t* output) ALWAYS_INLINE_LAMBDA {
        auto decodeNibbles = [&](simde_uint8x16_t v) ALWAYS_INLINE_LAMBDA {
            // Move digits '0'..'9' into range 0xf6..0xff.
            auto t1 = simde_vaddq_u8(v, SIMD::splat8(0xff - '9'));
            // And then correct the range to 0xf0..0xf9. All other bytes become less than 0xf0.
            auto t2 = simde_vqsubq_u8(t1, SIMD::splat8(6));
            // Convert '0'..'9' into nibbles 0..9. Non-digit bytes become greater than 0x0f.
            auto t3 = simde_vsubq_u8(t2, SIMD::splat8(0xf0));
            // Convert into uppercase 'a'..'f' => 'A'..'F'.
            auto t4 = simde_vandq_u8(v, SIMD::splat8(0xdf));
            // Move hex letter 'A'..'F' into range 0..5.
            auto t5 = simde_vsubq_u8(t4, SIMD::splat8('A'));
            // And correct the range into 10..15. The non-hex letters bytes become greater than 0x0f.
            auto t6 = simde_vqaddq_u8(t5, SIMD::splat8(10));
            // Finally choose the result: either valid nibble (0..9/10..15) or some byte greater than 0x0f.
            auto t7 = simde_vminq_u8(t3, t6);
            // Detect errors, i.e. bytes greater than 15.
            auto t8 = SIMD::greaterThan(t7, SIMD::splat8(15));
            return std::tuple { t7, t8 };
        };

        auto [nibbles, flags] = decodeNibbles(input);
        if (UNLIKELY(SIMD::isNonZero(flags)))
            return false;

        auto converted = simde_vreinterpretq_u16_u8(nibbles);
        auto low = simde_vshrq_n_u16(converted, 8);
        auto high = simde_vshlq_n_u16(converted, 4);
        simde_vst1_u8(output, simde_vmovn_u16(simde_vorrq_u16(low, high)));
        return true;
    };

    const auto* begin = span.data();
    const auto* end = span.data() + span.size();
    const auto* cursor = begin;

    auto* output = result.data();
    auto* outputEnd = result.data() + result.size();

    constexpr size_t stride = 16;
    constexpr size_t halfStride = stride / 2;
    if (span.size() >= stride) {
        auto doStridedDecode = [&]() ALWAYS_INLINE_LAMBDA {
            if constexpr (sizeof(CharacterType) == 1) {
                for (; cursor + stride <= end; cursor += stride, output += halfStride) {
                    if (!vectorDecode8(SIMD::load(std::bit_cast<const uint8_t*>(cursor)), output))
                        return false;
                }
                if (cursor < end) {
                    if (!vectorDecode8(SIMD::load(std::bit_cast<const uint8_t*>(end - stride)), outputEnd - halfStride))
                        return false;
                }
                return true;
            } else {
                auto vectorDecode16 = [&](simde_uint8x16x2_t input, uint8_t* output) ALWAYS_INLINE_LAMBDA {
                    if (UNLIKELY(SIMD::isNonZero(input.val[1])))
                        return false;
                    return vectorDecode8(input.val[0], output);
                };

                for (; cursor + stride <= end; cursor += stride, output += halfStride) {
                    if (!vectorDecode16(simde_vld2q_u8(std::bit_cast<const uint8_t*>(cursor)), output))
                        return false;
                }
                if (cursor < end) {
                    if (!vectorDecode16(simde_vld2q_u8(std::bit_cast<const uint8_t*>(end - stride)), outputEnd - halfStride))
                        return false;
                }
                return true;
            }
        };

        if (doStridedDecode())
            return WTF::notFound;
    }

    // 1. For small strings less than stride.
    // 2. Vector decoding failed due to incorrect character. Now, we do decoding character by character to decode up to the incorrect character position.
    for (; cursor < end; cursor += 2, output += 1) {
        int tens = parseDigit(*cursor, 16);
        if (UNLIKELY(tens == -1))
            return cursor - begin;
        int ones = parseDigit(*(cursor + 1), 16);
        if (UNLIKELY(ones == -1))
            return (cursor + 1) - begin;
        *output = (tens * 16) + ones;
    }
    return WTF::notFound;
}

WARN_UNUSED_RETURN size_t decodeHex(std::span<const LChar> span, std::span<uint8_t> result)
{
    return decodeHexImpl(span, result);
}

WARN_UNUSED_RETURN size_t decodeHex(std::span<const UChar> span, std::span<uint8_t> result)
{
    return decodeHexImpl(span, result);
}

JSC_DEFINE_HOST_FUNCTION(uint8ArrayConstructorFromHex, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSString* jsString = jsDynamicCast<JSString*>(callFrame->argument(0));
    if (UNLIKELY(!jsString))
        return throwVMTypeError(globalObject, scope, "Uint8Array.fromHex requires a string"_s);
    if (UNLIKELY(jsString->length() % 2))
        return JSValue::encode(throwSyntaxError(globalObject, scope, "Uint8Array.fromHex requires a string of even length"_s));

    StringView view = jsString->view(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    size_t count = static_cast<size_t>(view.length() / 2);
    JSUint8Array* uint8Array = JSUint8Array::createUninitialized(globalObject, globalObject->typedArrayStructure(TypeUint8, false), count);
    RETURN_IF_EXCEPTION(scope, { });

    uint8_t* data = uint8Array->typedVector();
    auto result = std::span { data, data + count };

    bool success = false;
    if (view.is8Bit())
        success = decodeHex(view.span8(), result) == WTF::notFound;
    else
        success = decodeHex(view.span16(), result) == WTF::notFound;

    if (UNLIKELY(!success))
        return JSValue::encode(throwSyntaxError(globalObject, scope, "Uint8Array.prototype.fromHex requires a string containing only \"0123456789abcdefABCDEF\""_s));

    return JSValue::encode(uint8Array);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
