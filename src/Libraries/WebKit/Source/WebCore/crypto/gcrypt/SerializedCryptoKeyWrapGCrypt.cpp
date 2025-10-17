/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#include "SerializedCryptoKeyWrap.h"

#include "NotImplemented.h"
#include "WrappedCryptoKey.h"

namespace WebCore {

std::optional<Vector<uint8_t>> defaultWebCryptoMasterKey()
{
    notImplemented();
    return std::nullopt;
}

// Initially these helper functions were intended to perform KEK wrapping and unwrapping,
// but this is not required anymore, despite the function names and the Mac implementation
// still indicating otherwise.
// See https://bugs.webkit.org/show_bug.cgi?id=173883 for more info.

bool wrapSerializedCryptoKey(const Vector<uint8_t>& masterKey, const Vector<uint8_t>& key, Vector<uint8_t>& result)
{
    UNUSED_PARAM(masterKey);

    // No wrapping performed -- the serialized key data is copied into the `result` variable.
    result = Vector<uint8_t>(key);
    return true;
}

std::optional<struct WrappedCryptoKey> readSerializedCryptoKey(const Vector<uint8_t>& wrappedKey)
{
    std::array<uint8_t, 24> a { 0 };
    std::array<uint8_t, 16> b { 0 };
    struct WrappedCryptoKey k { a, wrappedKey, b };
    return k;
}

std::optional<Vector<uint8_t>> unwrapCryptoKey([[maybe_unused]] const Vector<uint8_t>& masterKey, const struct WrappedCryptoKey& wrappedKey)
{
    return wrappedKey.encryptedKey;
}

} // namespace WebCore
