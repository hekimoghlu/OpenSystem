/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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

#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebDriver {

class HTTPParser {
public:
    struct Message {
        String path;
        String method;
        String version;
        Vector<String> requestHeaders;
        Vector<uint8_t> requestBody;
    };

    enum class Phase {
        Idle,
        Header,
        Body,
        Complete,
        Error,
    };

    enum class Process {
        Suspend,
        Continue,
    };

    Phase parse(Vector<uint8_t>&&);
    Message pullMessage() { return m_message; }

private:
    Process handlePhase();
    Process abortProcess(const char* message = nullptr);
    bool parseFirstLine(String&& line);
    bool readLine(String& line);
    size_t expectedBodyLength() const;

    Phase m_phase { Phase::Idle };
    Message m_message;
    size_t m_bodyLength { 0 };

    Vector<uint8_t> m_buffer;
};

} // namespace WebDriver
