/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#include "rtc_tools/rtc_event_log_visualizer/analyzer_common.h"

#include <cstdint>
#include <string>

#include "logging/rtc_event_log/rtc_event_log_parser.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

bool IsRtxSsrc(const ParsedRtcEventLog& parsed_log,
               PacketDirection direction,
               uint32_t ssrc) {
  if (direction == kIncomingPacket) {
    return parsed_log.incoming_rtx_ssrcs().find(ssrc) !=
           parsed_log.incoming_rtx_ssrcs().end();
  } else {
    return parsed_log.outgoing_rtx_ssrcs().find(ssrc) !=
           parsed_log.outgoing_rtx_ssrcs().end();
  }
}

bool IsVideoSsrc(const ParsedRtcEventLog& parsed_log,
                 PacketDirection direction,
                 uint32_t ssrc) {
  if (direction == kIncomingPacket) {
    return parsed_log.incoming_video_ssrcs().find(ssrc) !=
           parsed_log.incoming_video_ssrcs().end();
  } else {
    return parsed_log.outgoing_video_ssrcs().find(ssrc) !=
           parsed_log.outgoing_video_ssrcs().end();
  }
}

bool IsAudioSsrc(const ParsedRtcEventLog& parsed_log,
                 PacketDirection direction,
                 uint32_t ssrc) {
  if (direction == kIncomingPacket) {
    return parsed_log.incoming_audio_ssrcs().find(ssrc) !=
           parsed_log.incoming_audio_ssrcs().end();
  } else {
    return parsed_log.outgoing_audio_ssrcs().find(ssrc) !=
           parsed_log.outgoing_audio_ssrcs().end();
  }
}

std::string GetStreamName(const ParsedRtcEventLog& parsed_log,
                          PacketDirection direction,
                          uint32_t ssrc) {
  char buffer[200];
  rtc::SimpleStringBuilder name(buffer);
  if (IsAudioSsrc(parsed_log, direction, ssrc)) {
    name << "Audio ";
  } else if (IsVideoSsrc(parsed_log, direction, ssrc)) {
    name << "Video ";
  } else {
    name << "Unknown ";
  }
  if (IsRtxSsrc(parsed_log, direction, ssrc)) {
    name << "RTX ";
  }
  if (direction == kIncomingPacket)
    name << "(In) ";
  else
    name << "(Out) ";
  name << "SSRC " << ssrc;
  return name.str();
}

std::string GetLayerName(LayerDescription layer) {
  char buffer[100];
  rtc::SimpleStringBuilder name(buffer);
  name << "SSRC " << layer.ssrc << " sl " << layer.spatial_layer << ", tl "
       << layer.temporal_layer;
  return name.str();
}

}  // namespace webrtc
