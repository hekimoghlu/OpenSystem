/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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

#include <wtf/URL.h>

namespace IPC {
template<typename T, typename> struct ArgumentCoder;
}

namespace WebCore {

class BlobPart {
private:
    friend struct IPC::ArgumentCoder<BlobPart, void>;
public:
    enum class Type : bool {
        Data,
        Blob
    };

    BlobPart()
        : m_dataOrURL(Vector<uint8_t> { })
    {
    }

    BlobPart(Vector<uint8_t>&& data)
        : m_dataOrURL(WTFMove(data))
    {
    }

    BlobPart(const URL& url)
        : m_dataOrURL(url)
    {
    }

    Type type() const
    {
        return std::holds_alternative<URL>(m_dataOrURL) ? Type::Blob : Type::Data;
    }

    Vector<uint8_t>&& moveData()
    {
        ASSERT(std::holds_alternative<Vector<uint8_t>>(m_dataOrURL));
        return WTFMove(std::get<Vector<uint8_t>>(m_dataOrURL));
    }

    const URL& url() const
    {
        ASSERT(std::holds_alternative<URL>(m_dataOrURL));
        return std::get<URL>(m_dataOrURL);
    }

    void detachFromCurrentThread()
    {
        if (std::holds_alternative<URL>(m_dataOrURL))
            m_dataOrURL = std::get<URL>(m_dataOrURL).isolatedCopy();
    }

private:
    BlobPart(std::variant<Vector<uint8_t>, URL>&& dataOrURL)
        : m_dataOrURL(WTFMove(dataOrURL))
    {
    }

    std::variant<Vector<uint8_t>, URL> m_dataOrURL;
};

} // namespace WebCore
