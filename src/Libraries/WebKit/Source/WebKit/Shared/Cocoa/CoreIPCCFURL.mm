/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
#import "config.h"
#import "CoreIPCCFURL.h"

#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebKit {

std::optional<CoreIPCCFURL> CoreIPCCFURL::createWithBaseURLAndBytes(std::optional<CoreIPCCFURL>&& baseURL, Vector<uint8_t>&& bytes)
{
    if (bytes.isEmpty()) {
        // CFURL can't hold an empty URL, unlike NSURL.
        return CoreIPCCFURL { bridge_cast([NSURL URLWithString:@""]) };
    }

    CFURLRef baseCFURL = baseURL ? baseURL->m_cfURL.get() : nullptr;
    if (RetainPtr newCFURL = adoptCF(CFURLCreateAbsoluteURLWithBytes(nullptr, bytes.data(), bytes.size(), kCFStringEncodingUTF8, baseCFURL, true)))
        return CoreIPCCFURL { WTFMove(newCFURL) };

    return std::nullopt;
}

std::optional<CoreIPCCFURL> CoreIPCCFURL::baseURL() const
{
    if (CFURLRef baseURL = CFURLGetBaseURL(m_cfURL.get()))
        return CoreIPCCFURL { baseURL };
    return std::nullopt;
}

Vector<uint8_t> CoreIPCCFURL::toVector() const
{
    auto bytesLength = CFURLGetBytes(m_cfURL.get(), nullptr, 0);
    RELEASE_ASSERT(bytesLength != -1);
    Vector<uint8_t> result(bytesLength);
    CFURLGetBytes(m_cfURL.get(), result.data(), bytesLength);

    return result;
}

} // namespace WebKit

