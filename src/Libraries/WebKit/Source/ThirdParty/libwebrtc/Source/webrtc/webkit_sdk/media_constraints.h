/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
// Implementation of the w3c constraints spec is the responsibility of the
// browser. Chrome no longer uses the constraints api declared here, and it will
// be removed from WebRTC.
// https://bugs.chromium.org/p/webrtc/issues/detail?id=9239

#ifndef SDK_MEDIA_CONSTRAINTS_H_
#define SDK_MEDIA_CONSTRAINTS_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "api/audio_options.h"
#include "api/peer_connection_interface.h"

namespace webrtc {

// Class representing constraints, as used by the android and objc apis.
//
// Constraints may be either "mandatory", which means that unless satisfied,
// the method taking the constraints should fail, or "optional", which means
// they may not be satisfied..
class MediaConstraints {
 public:
  struct Constraint {
    Constraint() {}
    Constraint(const std::string& key, const std::string value)
        : key(key), value(value) {}
    std::string key;
    std::string value;
  };

  class Constraints : public std::vector<Constraint> {
   public:
    Constraints() = default;
    Constraints(std::initializer_list<Constraint> l)
        : std::vector<Constraint>(l) {}

    bool FindFirst(const std::string& key, std::string* value) const;
  };

  MediaConstraints() = default;
  MediaConstraints(Constraints mandatory, Constraints optional)
      : mandatory_(std::move(mandatory)), optional_(std::move(optional)) {}

  // Constraint keys used by a local audio source.

  // These keys are google specific.
  static const char kGoogEchoCancellation[];  // googEchoCancellation

  static const char kAutoGainControl[];   // googAutoGainControl
  static const char kNoiseSuppression[];  // googNoiseSuppression
  static const char kHighpassFilter[];    // googHighpassFilter
  static const char kAudioMirroring[];    // googAudioMirroring
  static const char
      kAudioNetworkAdaptorConfig[];  // googAudioNetworkAdaptorConfig
  static const char kInitAudioRecordingOnSend[];  // InitAudioRecordingOnSend;

  // Constraint keys for CreateOffer / CreateAnswer
  // Specified by the W3C PeerConnection spec
  static const char kOfferToReceiveVideo[];     // OfferToReceiveVideo
  static const char kOfferToReceiveAudio[];     // OfferToReceiveAudio
  static const char kVoiceActivityDetection[];  // VoiceActivityDetection
  static const char kIceRestart[];              // IceRestart
  // These keys are google specific.
  static const char kUseRtpMux[];  // googUseRtpMUX

  // Constraints values.
  static const char kValueTrue[];   // true
  static const char kValueFalse[];  // false

  // PeerConnection constraint keys.
  // Google-specific constraint keys.
  // Temporary pseudo-constraint for enabling DSCP through JS.
  static const char kEnableDscp[];  // googDscp
  // Constraint to enable IPv6 through JS.
  static const char kEnableIPv6[];  // googIPv6
  // Temporary constraint to enable suspend below min bitrate feature.
  static const char kEnableVideoSuspendBelowMinBitrate[];
  static const char kScreencastMinBitrate[];   // googScreencastMinBitrate
  static const char kCpuOveruseDetection[];    // googCpuOveruseDetection

  // Constraint to enable negotiating raw RTP packetization using attribute
  // "a=packetization:<payload_type> raw" in the SDP for all video payload.
  static const char kRawPacketizationForVideoEnabled[];

  // Specifies number of simulcast layers for all video tracks
  // with a Plan B offer/answer
  // (see RTCOfferAnswerOptions::num_simulcast_layers).
  static const char kNumSimulcastLayers[];

  ~MediaConstraints() = default;

  const Constraints& GetMandatory() const { return mandatory_; }
  const Constraints& GetOptional() const { return optional_; }

 private:
  const Constraints mandatory_ = {};
  const Constraints optional_ = {};
};

// Copy all relevant constraints into an RTCConfiguration object.
void CopyConstraintsIntoRtcConfiguration(
    const MediaConstraints* constraints,
    PeerConnectionInterface::RTCConfiguration* configuration);

// Copy all relevant constraints into an AudioOptions object.
void CopyConstraintsIntoAudioOptions(const MediaConstraints* constraints,
                                     cricket::AudioOptions* options);

bool CopyConstraintsIntoOfferAnswerOptions(
    const MediaConstraints* constraints,
    PeerConnectionInterface::RTCOfferAnswerOptions* offer_answer_options);

}  // namespace webrtc

#endif  // SDK_MEDIA_CONSTRAINTS_H_
