/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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

#include "Decoder.h"
#include "Encoder.h"
#include "SandboxExtension.h"
#include <WebCore/FormData.h>

namespace IPC {

class FormDataReference {
public:
    FormDataReference() = default;
    explicit FormDataReference(RefPtr<WebCore::FormData>&& data)
        : m_data(WTFMove(data))
    {
    }

    FormDataReference(RefPtr<WebCore::FormData>&&, Vector<WebKit::SandboxExtensionHandle>&&);

    RefPtr<WebCore::FormData> data() const { return m_data.get(); }
    RefPtr<WebCore::FormData> takeData() { return WTFMove(m_data); }

    Vector<WebKit::SandboxExtensionHandle> sandboxExtensionHandles() const;

private:
    RefPtr<WebCore::FormData> m_data;
};

inline FormDataReference::FormDataReference(RefPtr<WebCore::FormData>&& data, Vector<WebKit::SandboxExtensionHandle>&& sandboxExtensionHandles)
    : m_data(WTFMove(data))
{
    WebKit::SandboxExtension::consumePermanently(WTFMove(sandboxExtensionHandles));
}

inline Vector<WebKit::SandboxExtensionHandle> FormDataReference::sandboxExtensionHandles() const
{
    if (!m_data)
        return { };

    return WTF::compactMap(m_data->elements(), [](auto& element) -> std::optional<WebKit::SandboxExtensionHandle> {
        if (auto* fileData = std::get_if<WebCore::FormDataElement::EncodedFileData>(&element.data)) {
            const String& path = fileData->filename;
            if (auto handle = WebKit::SandboxExtension::createHandle(path, WebKit::SandboxExtension::Type::ReadOnly))
                return { WTFMove(*handle) };
        }
        return std::nullopt;
    });
}

} // namespace IPC
