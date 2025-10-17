/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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

#include "RTCDataChannelIdentifier.h"
#include "RTCDataChannelState.h"
#include "RTCErrorDetailType.h"
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCDataChannelRemoteSourceConnection : public ThreadSafeRefCounted<RTCDataChannelRemoteSourceConnection, WTF::DestructionThread::Main> {
public:
    virtual ~RTCDataChannelRemoteSourceConnection() = default;

    virtual void didChangeReadyState(RTCDataChannelIdentifier, RTCDataChannelState) = 0;
    virtual void didReceiveStringData(RTCDataChannelIdentifier, const String&) = 0;
    virtual void didReceiveRawData(RTCDataChannelIdentifier, std::span<const uint8_t>) = 0;
    virtual void didDetectError(RTCDataChannelIdentifier, RTCErrorDetailType, const String&) = 0;
    virtual void bufferedAmountIsDecreasing(RTCDataChannelIdentifier, size_t) = 0;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
