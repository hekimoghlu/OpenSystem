/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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

#include <gcrypt.h>
#include <optional>
#include <wtf/Assertions.h>

namespace PAL {
namespace GCrypt {

// Copied from gcrypt.h. This macro is new in version 1.7.0, which we do not
// want to require yet. Remove this in summer 2019.
#ifndef gcry_cipher_final
#define gcry_cipher_final(a) \
            gcry_cipher_ctl ((a), GCRYCTL_FINALIZE, NULL, 0)
#endif

using CipherOperation = gcry_error_t(gcry_cipher_hd_t, void*, size_t, const void*, size_t);

static inline void logError(gcry_error_t error)
{
    // FIXME: Use a WebCrypto WTF log channel here once those are moved down to PAL.
#if !LOG_DISABLED
    WTFLogAlways("libgcrypt error: source '%s', description '%s'",
        gcry_strsource(error), gcry_strerror(error));
#else
    UNUSED_PARAM(error);
#endif
}

static inline std::optional<int> aesAlgorithmForKeySize(size_t keySize)
{
    switch (keySize) {
    case 128:
        return GCRY_CIPHER_AES128;
    case 192:
        return GCRY_CIPHER_AES192;
    case 256:
        return GCRY_CIPHER_AES256;
    default:
        return std::nullopt;
    }
}

} // namespace GCrypt
} // namespace PAL
