/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#include "JSRTCRtpSFrameTransform.h"

#if ENABLE(WEB_RTC)

#include "JSCryptoKey.h"
#include "JSDOMPromiseDeferred.h"
#include <JavaScriptCore/JSBigInt.h>

namespace WebCore {
using namespace JSC;

JSValue JSRTCRtpSFrameTransform::setEncryptionKey(JSGlobalObject& lexicalGlobalObject, CallFrame& callFrame, Ref<DeferredPromise>&& promise)
{
    auto& vm = getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(callFrame.argumentCount() < 1)) {
        throwVMError(&lexicalGlobalObject, throwScope, createNotEnoughArgumentsError(&lexicalGlobalObject));
        return jsUndefined();
    }

    EnsureStillAliveScope argument0 = callFrame.uncheckedArgument(0);
    auto keyConversionResult = convert<IDLInterface<CryptoKey>>(lexicalGlobalObject, argument0.value(), [](auto& lexicalGlobalObject, auto& scope) {
        throwArgumentTypeError(lexicalGlobalObject, scope, 0, "key"_s, "SFrameTransform"_s, "setEncryptionKey"_s, "CryptoKey"_s);
    });
    if (UNLIKELY(keyConversionResult.hasException(throwScope)))
        return jsUndefined();

    EnsureStillAliveScope argument1 = callFrame.argument(1);
    std::optional<uint64_t> keyID;
    if (!argument1.value().isUndefined()) {
        if (argument1.value().isBigInt()) {
            if (argument1.value().asHeapBigInt()->length() > 1) {
                throwException(&lexicalGlobalObject, throwScope, createDOMException(&lexicalGlobalObject, ExceptionCode::RangeError, "Not a 64 bits integer"_s));
                return jsUndefined();
            }
            keyID = JSBigInt::toBigUInt64(argument1.value());
        } else {
            auto keyIDConversionResult = convert<IDLUnsignedLongLong>(lexicalGlobalObject, argument1.value());
            if (UNLIKELY(keyIDConversionResult.hasException(throwScope)))
                return jsUndefined();
            keyID = keyIDConversionResult.releaseReturnValue();
        }
    }
    RETURN_IF_EXCEPTION(throwScope, jsUndefined());
    throwScope.release();

    wrapped().setEncryptionKey(*keyConversionResult.releaseReturnValue(), keyID, WTFMove(promise));
    return jsUndefined();
}

}

#endif
