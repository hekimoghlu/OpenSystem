/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#include "ShareableResource.h"
#include "SharedBuffer.h"

namespace WebCore {

class ScriptBuffer {
public:
    ScriptBuffer() = default;

    ScriptBuffer(RefPtr<FragmentedSharedBuffer>&& buffer)
        : m_buffer(WTFMove(buffer))
    {
    }

    static ScriptBuffer empty();

    WEBCORE_EXPORT explicit ScriptBuffer(const String&);

    String toString() const;
    const FragmentedSharedBuffer* buffer() const { return m_buffer.get().get(); }

    ScriptBuffer isolatedCopy() const { return ScriptBuffer(m_buffer ? RefPtr<FragmentedSharedBuffer>(m_buffer.copy()) : nullptr); }
    explicit operator bool() const { return !!m_buffer; }
    bool isEmpty() const { return m_buffer.isEmpty(); }

    WEBCORE_EXPORT bool containsSingleFileMappedSegment() const;
    void append(const String&);
    void append(const FragmentedSharedBuffer&);

#if ENABLE(SHAREABLE_RESOURCE) && PLATFORM(COCOA)
    using IPCData = std::variant<ShareableResourceHandle, RefPtr<FragmentedSharedBuffer>>;
#else
    using IPCData = RefPtr<FragmentedSharedBuffer>;
#endif

    WEBCORE_EXPORT static std::optional<ScriptBuffer> fromIPCData(IPCData&&);
    WEBCORE_EXPORT IPCData ipcData() const;

private:
    SharedBufferBuilder m_buffer; // Contains the UTF-8 encoded script.
};

bool operator==(const ScriptBuffer&, const ScriptBuffer&);

} // namespace WebCore
