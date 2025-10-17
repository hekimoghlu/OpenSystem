/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#ifndef MODULES_AUDIO_CODING_CODECS_OPUS_OPUS_INST_H_
#define MODULES_AUDIO_CODING_CODECS_OPUS_OPUS_INST_H_

#include <stddef.h>

#include "rtc_base/ignore_wundef.h"

RTC_PUSH_IGNORING_WUNDEF()
#include "third_party/opus/src/include/opus.h"
#include "third_party/opus/src/include/opus_multistream.h"
RTC_POP_IGNORING_WUNDEF()

struct WebRtcOpusEncInst {
  OpusEncoder* encoder;
  OpusMSEncoder* multistream_encoder;
  size_t channels;
  int in_dtx_mode;
  int sample_rate_hz;
};

struct WebRtcOpusDecInst {
  OpusDecoder* decoder;
  OpusMSDecoder* multistream_decoder;
  size_t channels;
  int in_dtx_mode;
  int sample_rate_hz;
  // TODO: https://issues.webrtc.org/376493209 - Remove when libopus gets fixed.
  int last_packet_num_channels;
};

#endif  // MODULES_AUDIO_CODING_CODECS_OPUS_OPUS_INST_H_
