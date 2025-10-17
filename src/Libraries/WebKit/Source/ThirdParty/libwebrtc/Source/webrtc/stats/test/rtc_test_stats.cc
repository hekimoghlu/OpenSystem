/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "stats/test/rtc_test_stats.h"

#include "api/stats/attribute.h"
#include "rtc_base/checks.h"

namespace webrtc {

WEBRTC_RTCSTATS_IMPL(RTCTestStats,
                     RTCStats,
                     "test-stats",
                     AttributeInit("mBool", &m_bool),
                     AttributeInit("mInt32", &m_int32),
                     AttributeInit("mUint32", &m_uint32),
                     AttributeInit("mInt64", &m_int64),
                     AttributeInit("mUint64", &m_uint64),
                     AttributeInit("mDouble", &m_double),
                     AttributeInit("mString", &m_string),
                     AttributeInit("mSequenceBool", &m_sequence_bool),
                     AttributeInit("mSequenceInt32", &m_sequence_int32),
                     AttributeInit("mSequenceUint32", &m_sequence_uint32),
                     AttributeInit("mSequenceInt64", &m_sequence_int64),
                     AttributeInit("mSequenceUint64", &m_sequence_uint64),
                     AttributeInit("mSequenceDouble", &m_sequence_double),
                     AttributeInit("mSequenceString", &m_sequence_string),
                     AttributeInit("mMapStringUint64", &m_map_string_uint64),
                     AttributeInit("mMapStringDouble", &m_map_string_double))

RTCTestStats::RTCTestStats(const std::string& id, Timestamp timestamp)
    : RTCStats(id, timestamp) {}

RTCTestStats::~RTCTestStats() {}

}  // namespace webrtc
