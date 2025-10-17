/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefPtr.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

// See Bug 274508: Disable thread-safety-reference-return warnings in libwebrtc
IGNORE_CLANG_WARNINGS_BEGIN("thread-safety-reference-return")
#include <webrtc/pc/rtc_stats_collector.h>
IGNORE_CLANG_WARNINGS_END

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

class DOMMapAdapter;
class RTCStatsReport;

void initializeRTCStatsReportBackingMap(RTCStatsReport&);

class LibWebRTCStatsCollector : public webrtc::RTCStatsCollectorCallback {
public:
    using CollectorCallback = CompletionHandler<void(const rtc::scoped_refptr<const webrtc::RTCStatsReport>&)>;
    static rtc::scoped_refptr<LibWebRTCStatsCollector> create(CollectorCallback&& callback) { return rtc::make_ref_counted<LibWebRTCStatsCollector>(WTFMove(callback)); }

    static Ref<RTCStatsReport> createReport(const rtc::scoped_refptr<const webrtc::RTCStatsReport>&);

    explicit LibWebRTCStatsCollector(CollectorCallback&&);
    ~LibWebRTCStatsCollector();

private:
    void OnStatsDelivered(const rtc::scoped_refptr<const webrtc::RTCStatsReport>&) final;

    CollectorCallback m_callback;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
