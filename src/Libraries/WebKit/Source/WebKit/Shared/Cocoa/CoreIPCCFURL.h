/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#if PLATFORM(COCOA)

#include <wtf/ArgumentCoder.h>
#include <wtf/URL.h>

namespace WebKit {

class CoreIPCCFURL {
public:
    CoreIPCCFURL(CFURLRef url)
        : m_cfURL(url)
    {
        RELEASE_ASSERT(url);
    }

    CoreIPCCFURL(RetainPtr<CFURLRef> url)
        : m_cfURL(WTFMove(url))
    {
        RELEASE_ASSERT(m_cfURL);
    }

    std::optional<CoreIPCCFURL> baseURL() const;
    Vector<uint8_t> toVector() const;

    RetainPtr<CFURLRef> createCFURL() const { return m_cfURL; }

    static std::optional<CoreIPCCFURL> createWithBaseURLAndBytes(std::optional<CoreIPCCFURL>&& baseURL, Vector<uint8_t>&& bytes);

private:
    RetainPtr<CFURLRef> m_cfURL;
};

} // namespace WebKit

#endif // PLATFORM(COCOA)
