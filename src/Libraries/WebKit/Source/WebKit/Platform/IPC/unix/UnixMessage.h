/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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

#include "Attachment.h"
#include "Encoder.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace IPC {

class MessageInfo {
public:
    MessageInfo()
    {
        // The entire MessageInfo is passed to write(), so we have to zero our
        // padding bytes to avoid writing uninitialized memory.
        memset(static_cast<void*>(this), 0, sizeof(*this));
    }

    MessageInfo(size_t bodySize, size_t initialAttachmentCount)
    {
        memset(static_cast<void*>(this), 0, sizeof(*this));
        m_bodySize = bodySize;
        m_attachmentCount = initialAttachmentCount;
    }

    MessageInfo(const MessageInfo& info)
    {
        memset(static_cast<void*>(this), 0, sizeof(*this));
        *this = info;
    }

    MessageInfo& operator=(const MessageInfo&) = default;

    void setBodyOutOfLine()
    {
        ASSERT(!isBodyOutOfLine());

        m_isBodyOutOfLine = true;
        m_attachmentCount++;
    }

    bool isBodyOutOfLine() const { return m_isBodyOutOfLine; }
    size_t bodySize() const { return m_bodySize; }
    size_t attachmentCount() const { return m_attachmentCount; }

private:
    // The MessageInfo will be copied using memcpy, so all members must be trivially copyable.
    size_t m_bodySize;
    size_t m_attachmentCount;
    bool m_isBodyOutOfLine;
};

class UnixMessage {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(UnixMessage);
public:
    UnixMessage(Encoder& encoder)
        : m_attachments(encoder.releaseAttachments())
        , m_messageInfo(encoder.span().size(), m_attachments.size())
        , m_body(const_cast<uint8_t*>(encoder.span().data()), encoder.span().size())
    {
    }

    UnixMessage(UnixMessage&& other)
        : m_attachments(WTFMove(other.m_attachments))
        , m_messageInfo(WTFMove(other.m_messageInfo))
    {
        if (other.m_bodyOwned) {
            std::swap(m_body, other.m_body);
            std::swap(m_bodyOwned, other.m_bodyOwned);
        } else if (!m_messageInfo.isBodyOutOfLine()) {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Unix port
            m_body = std::span { static_cast<uint8_t*>(fastMalloc(m_messageInfo.bodySize())), m_messageInfo.bodySize() };
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
            memcpySpan(m_body, other.m_body);
            m_bodyOwned = true;
            other.m_body = { };
            other.m_bodyOwned = false;
        }
    }

    ~UnixMessage()
    {
        if (m_bodyOwned)
            fastFree(m_body.data());
    }

    const Vector<Attachment>& attachments() const { return m_attachments; }
    MessageInfo& messageInfo() { return m_messageInfo; }

    std::span<uint8_t> body() const { return m_body; }
    size_t bodySize() const  { return m_messageInfo.bodySize(); }

    void appendAttachment(Attachment&& attachment)
    {
        m_attachments.append(WTFMove(attachment));
    }

private:
    Vector<Attachment> m_attachments;
    MessageInfo m_messageInfo;
    std::span<uint8_t> m_body;
    bool m_bodyOwned { false };
};

} // namespace IPC
