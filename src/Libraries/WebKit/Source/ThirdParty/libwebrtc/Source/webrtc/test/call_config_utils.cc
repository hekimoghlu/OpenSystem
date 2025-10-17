/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#include "test/call_config_utils.h"

#include <string>
#include <vector>

namespace webrtc {
namespace test {

// Deserializes a JSON representation of the VideoReceiveStreamInterface::Config
// back into a valid object. This will not initialize the decoders or the
// renderer.
VideoReceiveStreamInterface::Config ParseVideoReceiveStreamJsonConfig(
    webrtc::Transport* transport,
    const Json::Value& json) {
  auto receive_config = VideoReceiveStreamInterface::Config(transport);
  for (const auto& decoder_json : json["decoders"]) {
    VideoReceiveStreamInterface::Decoder decoder;
#if WEBRTC_WEBKIT_BUILD
    decoder.video_format =
        SdpVideoFormat(decoder_json["payload_name"].get<std::string>());
    decoder.payload_type = decoder_json["payload_type"].get<int32_t>();
#else
    decoder.video_format =
        SdpVideoFormat(decoder_json["payload_name"].asString());
    decoder.payload_type = decoder_json["payload_type"].asInt64();
#endif
    for (const auto& params_json : decoder_json["codec_params"]) {
#if WEBRTC_WEBKIT_BUILD
      std::vector<std::string> members(params_json.size());
      for (const auto& [key, value] : params_json.items()) {
        members.emplace_back(key);
      }
      RTC_CHECK_EQ(members.size(), 1);
      decoder.video_format.parameters[members[0]] =
          params_json[members[0]].get<std::string>();
#else
      std::vector<std::string> members = params_json.getMemberNames();
      RTC_CHECK_EQ(members.size(), 1);
      decoder.video_format.parameters[members[0]] =
          params_json[members[0]].asString();
#endif
    }
    receive_config.decoders.push_back(decoder);
  }
#if WEBRTC_WEBKIT_BUILD
  receive_config.render_delay_ms = json["render_delay_ms"].get<int32_t>();
  receive_config.rtp.remote_ssrc = json["rtp"]["remote_ssrc"].get<uint32_t>();
  receive_config.rtp.local_ssrc = json["rtp"]["local_ssrc"].get<uint32_t>();
  receive_config.rtp.rtcp_mode =
      json["rtp"]["rtcp_mode"].get<std::string>() == "RtcpMode::kCompound"
          ? RtcpMode::kCompound
          : RtcpMode::kReducedSize;
  receive_config.rtp.lntf.enabled = json["rtp"]["lntf"]["enabled"].get<bool>();
  receive_config.rtp.nack.rtp_history_ms =
      json["rtp"]["nack"]["rtp_history_ms"].get<int32_t>();
  receive_config.rtp.ulpfec_payload_type =
      json["rtp"]["ulpfec_payload_type"].get<int32_t>();
  receive_config.rtp.red_payload_type =
      json["rtp"]["red_payload_type"].get<int32_t>();
  receive_config.rtp.rtx_ssrc = json["rtp"]["rtx_ssrc"].get<uint32_t>();
#else
  receive_config.render_delay_ms = json["render_delay_ms"].asInt64();
  receive_config.rtp.remote_ssrc = json["rtp"]["remote_ssrc"].asInt64();
  receive_config.rtp.local_ssrc = json["rtp"]["local_ssrc"].asInt64();
  receive_config.rtp.rtcp_mode =
      json["rtp"]["rtcp_mode"].asString() == "RtcpMode::kCompound"
          ? RtcpMode::kCompound
          : RtcpMode::kReducedSize;
  receive_config.rtp.lntf.enabled = json["rtp"]["lntf"]["enabled"].asInt64();
  receive_config.rtp.nack.rtp_history_ms =
      json["rtp"]["nack"]["rtp_history_ms"].asInt64();
  receive_config.rtp.ulpfec_payload_type =
      json["rtp"]["ulpfec_payload_type"].asInt64();
  receive_config.rtp.red_payload_type =
      json["rtp"]["red_payload_type"].asInt64();
  receive_config.rtp.rtx_ssrc = json["rtp"]["rtx_ssrc"].asInt64();
#endif

  for (const auto& pl_json : json["rtp"]["rtx_payload_types"]) {
#if WEBRTC_WEBKIT_BUILD
    std::vector<std::string> members(pl_json.size());
    for (const auto& [key, value] : pl_json.items()) {
        members.emplace_back(key);
    }
    RTC_CHECK_EQ(members.size(), 1);
    Json::Value rtx_payload_type = pl_json[members[0]];
    receive_config.rtp.rtx_associated_payload_types[std::stoi(members[0])] =
        rtx_payload_type.get<int32_t>();
#else
    std::vector<std::string> members = pl_json.getMemberNames();
    RTC_CHECK_EQ(members.size(), 1);
    Json::Value rtx_payload_type = pl_json[members[0]];
    receive_config.rtp.rtx_associated_payload_types[std::stoi(members[0])] =
        rtx_payload_type.asInt64();
#endif
  }
  return receive_config;
}

Json::Value GenerateVideoReceiveStreamJsonConfig(
    const VideoReceiveStreamInterface::Config& config) {
  Json::Value root_json;

#if WEBRTC_WEBKIT_BUILD
  root_json["decoders"] = Json::Value("[]");
#else
  root_json["decoders"] = Json::Value(Json::arrayValue);
#endif
  for (const auto& decoder : config.decoders) {
    Json::Value decoder_json;
    decoder_json["payload_type"] = decoder.payload_type;
    decoder_json["payload_name"] = decoder.video_format.name;
#if WEBRTC_WEBKIT_BUILD
    decoder_json["codec_params"] = Json::Value("[]");
#else
    decoder_json["codec_params"] = Json::Value(Json::arrayValue);
#endif
    for (const auto& codec_param_entry : decoder.video_format.parameters) {
      Json::Value codec_param_json;
      codec_param_json[codec_param_entry.first] = codec_param_entry.second;
#if WEBRTC_WEBKIT_BUILD
      decoder_json["codec_params"].push_back(codec_param_json);
#else
      decoder_json["codec_params"].append(codec_param_json);
#endif
    }
#if WEBRTC_WEBKIT_BUILD
    root_json["decoders"].push_back(decoder_json);
#else
    root_json["decoders"].append(decoder_json);
#endif
  }

  Json::Value rtp_json;
  rtp_json["remote_ssrc"] = config.rtp.remote_ssrc;
  rtp_json["local_ssrc"] = config.rtp.local_ssrc;
  rtp_json["rtcp_mode"] = config.rtp.rtcp_mode == RtcpMode::kCompound
                              ? "RtcpMode::kCompound"
                              : "RtcpMode::kReducedSize";
  rtp_json["lntf"]["enabled"] = config.rtp.lntf.enabled;
  rtp_json["nack"]["rtp_history_ms"] = config.rtp.nack.rtp_history_ms;
  rtp_json["ulpfec_payload_type"] = config.rtp.ulpfec_payload_type;
  rtp_json["red_payload_type"] = config.rtp.red_payload_type;
  rtp_json["rtx_ssrc"] = config.rtp.rtx_ssrc;
#if WEBRTC_WEBKIT_BUILD
  rtp_json["rtx_payload_types"] = Json::Value("[]");
#else
  rtp_json["rtx_payload_types"] = Json::Value(Json::arrayValue);
#endif

  for (auto& kv : config.rtp.rtx_associated_payload_types) {
    Json::Value val;
    val[std::to_string(kv.first)] = kv.second;
#if WEBRTC_WEBKIT_BUILD
    rtp_json["rtx_payload_types"].push_back(val);
#else
    rtp_json["rtx_payload_types"].append(val);
#endif
  }

  root_json["rtp"] = rtp_json;

  root_json["render_delay_ms"] = config.render_delay_ms;

  return root_json;
}

}  // namespace test.
}  // namespace webrtc.
