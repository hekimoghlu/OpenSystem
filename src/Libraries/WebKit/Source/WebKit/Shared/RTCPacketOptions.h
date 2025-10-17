/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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

#if USE(LIBWEBRTC)

#include <WebCore/LibWebRTCMacros.h>
#include <wtf/Forward.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/rtc_base/async_packet_socket.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebKit {

struct RTCPacketOptions {
    enum class DifferentiatedServicesCodePoint : int8_t {
        NoChange = -1,
        Default = 0, // Same as CS0.
        CS0 = 0, // The default.
        CS1 = 8, // Bulk/background traffic.
        AF11 = 10,
        AF12 = 12,
        AF13 = 14,
        CS2 = 16,
        AF21 = 18,
        AF22 = 20,
        AF23 = 22,
        CS3 = 24,
        AF31 = 26,
        AF32 = 28,
        AF33 = 30,
        CS4 = 32,
        AF41 = 34, // Video.
        AF42 = 36, // Video.
        AF43 = 38, // Video.
        CS5 = 40, // Video.
        EF = 46, // Voice.
        CS6 = 48, // Voice.
        CS7 = 56 // Control messages.
    };

    struct SerializableData {
        DifferentiatedServicesCodePoint dscp;
        int32_t packetId;
        int rtpSendtimeExtensionId;
        int64_t srtpAuthTagLength;
        std::span<const char> srtpAuthKey;
        int64_t srtpPacketIndex;
    };

    explicit RTCPacketOptions(const rtc::PacketOptions& options)
        : options(options)
    { }

    explicit RTCPacketOptions(const SerializableData&);

    SerializableData serializableData() const;

    rtc::PacketOptions options;
};

}

#endif // USE(LIBWEBRTC)
