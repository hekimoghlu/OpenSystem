/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#include <wtf/SafeStrerror.h>

#include <cstring>
#include <type_traits>
#include <wtf/Platform.h>
#include <wtf/text/CString.h>

namespace WTF {

CString safeStrerror(int errnum)
{
    constexpr size_t bufferLength = 1024;
    std::span<char> cstringBuffer;
    auto result = CString::newUninitialized(bufferLength, cstringBuffer);

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#if OS(WINDOWS)
    strerror_s(cstringBuffer.data(), cstringBuffer.size(), errnum);
#else
    auto ret = strerror_r(errnum, cstringBuffer.data(), cstringBuffer.size());

    if constexpr (std::is_same<decltype(ret), char*>::value) {
        // We have GNU strerror_r(), which returns char*. This may or may not be a pointer into
        // cstringBuffer. We also have to be careful because this has to compile even if ret is
        // an int, hence the reinterpret_casts.
        char* message = reinterpret_cast<char*>(ret);
        if (message != cstringBuffer.data())
            strncpy(cstringBuffer.data(), message, cstringBuffer.size());
    } else {
        // We have POSIX strerror_r, which returns int and may fail.
        if (ret)
            snprintf(cstringBuffer.data(), cstringBuffer.size(), "%s %d", "Unknown error", errnum);
    }
#endif // OS(WINDOWS)

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    return result;
}

} // namespace WTF
