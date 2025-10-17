/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#include <wtf/RandomDevice.h>

#include <stdlib.h>

#if !OS(DARWIN) && !OS(FUCHSIA) && OS(UNIX)
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#if OS(WINDOWS)
#include <windows.h>
#include <wincrypt.h> // windows.h must be included before wincrypt.h.
#endif

#if OS(DARWIN)
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonRandom.h>
#endif

#if OS(FUCHSIA)
#include <zircon/syscalls.h>
#endif

namespace WTF {

#if !OS(DARWIN) && !OS(FUCHSIA) && OS(UNIX)
NEVER_INLINE NO_RETURN_DUE_TO_CRASH static void crashUnableToOpenURandom()
{
    CRASH();
}

NEVER_INLINE NO_RETURN_DUE_TO_CRASH static void crashUnableToReadFromURandom()
{
    CRASH();
}
#endif

#if !OS(DARWIN) && !OS(FUCHSIA) && !OS(WINDOWS)
RandomDevice::RandomDevice()
{
    int ret = 0;
    do {
        ret = open("/dev/urandom", O_RDONLY, 0);
    } while (ret == -1 && errno == EINTR);
    m_fd = ret;
    if (m_fd < 0)
        crashUnableToOpenURandom(); // We need /dev/urandom for this API to work...
}
#endif

#if !OS(DARWIN) && !OS(FUCHSIA) && !OS(WINDOWS)
RandomDevice::~RandomDevice()
{
    close(m_fd);
}
#endif

// FIXME: Make this call fast by creating the pool in RandomDevice.
// https://bugs.webkit.org/show_bug.cgi?id=170190
void RandomDevice::cryptographicallyRandomValues(std::span<uint8_t> buffer)
{
#if OS(DARWIN)
    RELEASE_ASSERT(!CCRandomGenerateBytes(buffer.data(), buffer.size()));
#elif OS(FUCHSIA)
    zx_cprng_draw(buffer.data(), buffer.size());
#elif OS(UNIX)
    ssize_t amountRead = 0;
    while (static_cast<size_t>(amountRead) < buffer.size()) {
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        ssize_t currentRead = read(m_fd, buffer.data() + amountRead, buffer.size() - amountRead);
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        // We need to check for both EAGAIN and EINTR since on some systems /dev/urandom
        // is blocking and on others it is non-blocking.
        if (currentRead == -1) {
            if (!(errno == EAGAIN || errno == EINTR))
                crashUnableToReadFromURandom();
        } else
            amountRead += currentRead;
    }
#elif OS(WINDOWS)
    // FIXME: We cannot ensure that Cryptographic Service Provider context and CryptGenRandom are safe across threads.
    // If it is safe, we can acquire context per RandomDevice.
    HCRYPTPROV hCryptProv = 0;
    if (!CryptAcquireContext(&hCryptProv, nullptr, MS_DEF_PROV, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
        CRASH();
    if (!CryptGenRandom(hCryptProv, buffer.size(), buffer.data()))
        CRASH();
    CryptReleaseContext(hCryptProv, 0);
#else
#error "This configuration doesn't have a strong source of randomness."
// WARNING: When adding new sources of OS randomness, the randomness must
//          be of cryptographic quality!
#endif
}

}
