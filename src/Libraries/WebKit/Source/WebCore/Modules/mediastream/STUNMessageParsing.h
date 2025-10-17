/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

#include <wtf/Function.h>
#include <wtf/Vector.h>

#if ENABLE(WEB_RTC)

namespace WebCore {
namespace WebRTC {

struct STUNMessageLengths {
    size_t messageLength { 0 };
    size_t messageLengthWithPadding { 0 };
};

WEBCORE_EXPORT std::optional<STUNMessageLengths> getSTUNOrTURNMessageLengths(std::span<const uint8_t>);

enum class MessageType { STUN, Data };
WEBCORE_EXPORT Vector<uint8_t> extractMessages(Vector<uint8_t>&&, MessageType, const Function<void(std::span<const uint8_t> data)>&);

} // namespace WebRTC
} // namespace WebCore

#endif
