/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#pragma once

namespace WebCore {

enum {
    CryptoKeyUsageEncrypt = 1 << 0,
    CryptoKeyUsageDecrypt = 1 << 1,
    CryptoKeyUsageSign = 1 << 2,
    CryptoKeyUsageVerify = 1 << 3,
    CryptoKeyUsageDeriveKey = 1 << 4,
    CryptoKeyUsageDeriveBits = 1 << 5,
    CryptoKeyUsageWrapKey = 1 << 6,
    CryptoKeyUsageUnwrapKey = 1 << 7
};

typedef int CryptoKeyUsageBitmap;

// Only for binding purpose.
enum class CryptoKeyUsage : uint8_t {
    Encrypt,
    Decrypt,
    Sign,
    Verify,
    DeriveKey,
    DeriveBits,
    WrapKey,
    UnwrapKey
};

} // namespace WebCore
