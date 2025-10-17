/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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

#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

class RandomDevice {
    WTF_MAKE_NONCOPYABLE(RandomDevice);
    WTF_MAKE_FAST_ALLOCATED;
public:
#if OS(DARWIN) || OS(FUCHSIA) || OS(WINDOWS)
    RandomDevice() = default;
#else
    RandomDevice();
    ~RandomDevice();
#endif

    // This function attempts to fill buffer with randomness from the operating
    // system. Rather than calling this function directly, consider calling
    // cryptographicallyRandomNumber or cryptographicallyRandomValues.
    void cryptographicallyRandomValues(std::span<uint8_t> buffer);

private:
#if OS(DARWIN) || OS(FUCHSIA) || OS(WINDOWS)
#elif OS(UNIX)
    int m_fd { -1 };
#else
#error "This configuration doesn't have a strong source of randomness."
// WARNING: When adding new sources of OS randomness, the randomness must
//          be of cryptographic quality!
#endif
};

}
