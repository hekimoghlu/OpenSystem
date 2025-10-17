/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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

#if ENABLE(REMOTE_INSPECTOR)

#include <wtf/Function.h>
#include <wtf/Vector.h>

namespace Inspector {

class MessageParser {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Vector<uint8_t> createMessage(std::span<const uint8_t>);

    MessageParser() { }
    MessageParser(Function<void(Vector<uint8_t>&&)>&&);
    void pushReceivedData(std::span<const uint8_t>);
    void clearReceivedData();

private:
    bool parse();

    Function<void(Vector<uint8_t>&&)> m_listener { };
    Vector<uint8_t> m_buffer;
};

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
