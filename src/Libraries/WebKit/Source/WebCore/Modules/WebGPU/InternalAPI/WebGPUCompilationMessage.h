/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#include "WebGPUCompilationMessageType.h"
#include <cstdint>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class CompilationMessage final : public RefCounted<CompilationMessage> {
public:
    static Ref<CompilationMessage> create(String&& message, CompilationMessageType type, uint64_t lineNum, uint64_t linePos, uint64_t offset, uint64_t length)
    {
        return adoptRef(*new CompilationMessage(WTFMove(message), type, lineNum, linePos, offset, length));
    }

    static Ref<CompilationMessage> create(const String& message, CompilationMessageType type, uint64_t lineNum, uint64_t linePos, uint64_t offset, uint64_t length)
    {
        return adoptRef(*new CompilationMessage(message, type, lineNum, linePos, offset, length));
    }

    const String& message() const { return m_message; }
    CompilationMessageType type() const { return m_type; }
    uint64_t lineNum() const { return m_lineNum; }
    uint64_t linePos() const { return m_linePos; }
    uint64_t offset() const { return m_offset; }
    uint64_t length() const { return m_length; }

private:
    CompilationMessage(String&& message, CompilationMessageType type, uint64_t lineNum, uint64_t linePos, uint64_t offset, uint64_t length)
        : m_message(WTFMove(message))
        , m_type(type)
        , m_lineNum(lineNum)
        , m_linePos(linePos)
        , m_offset(offset)
        , m_length(length)
    {
    }

    CompilationMessage(const String& message, CompilationMessageType type, uint64_t lineNum, uint64_t linePos, uint64_t offset, uint64_t length)
        : m_message(message)
        , m_type(type)
        , m_lineNum(lineNum)
        , m_linePos(linePos)
        , m_offset(offset)
        , m_length(length)
    {
    }

    CompilationMessage(const CompilationMessage&) = delete;
    CompilationMessage(CompilationMessage&&) = delete;
    CompilationMessage& operator=(const CompilationMessage&) = delete;
    CompilationMessage& operator=(CompilationMessage&&) = delete;

    String m_message;
    CompilationMessageType m_type { CompilationMessageType::Error };
    uint64_t m_lineNum { 0 };
    uint64_t m_linePos { 0 };
    uint64_t m_offset { 0 };
    uint64_t m_length { 0 };
};

} // namespace WebCore::WebGPU
