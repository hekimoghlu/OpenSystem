/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

#if ENABLE(WEB_RTC)

#include "DOMException.h"
#include "RTCErrorDetailType.h"
#include <optional>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCError final : public DOMException {
public:
    struct Init {
        RTCErrorDetailType errorDetail;
        std::optional<int> sdpLineNumber;
        std::optional<int> sctpCauseCode;
        std::optional<unsigned> receivedAlert;
        std::optional<unsigned> sentAlert;
    };

    static Ref<RTCError> create(const Init& init, String&& message) { return adoptRef(*new RTCError(init, WTFMove(message))); }
    static Ref<RTCError> create(RTCErrorDetailType type, String&& message) { return create({ type, { }, { }, { }, { } }, WTFMove(message)); }

    RTCErrorDetailType errorDetail() const { return m_values.errorDetail; }
    std::optional<int> sdpLineNumber() const  { return m_values.sdpLineNumber; }
    std::optional<int> sctpCauseCode() const  { return m_values.sctpCauseCode; }
    std::optional<unsigned> receivedAlert() const  { return m_values.receivedAlert; }
    std::optional<unsigned> sentAlert() const  { return m_values.sentAlert; }

private:
    WEBCORE_EXPORT RTCError(const Init&, String&&);

    Init m_values;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
