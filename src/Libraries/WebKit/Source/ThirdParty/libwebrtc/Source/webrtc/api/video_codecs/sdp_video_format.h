/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#ifndef API_VIDEO_CODECS_SDP_VIDEO_FORMAT_H_
#define API_VIDEO_CODECS_SDP_VIDEO_FORMAT_H_

#include <map>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "api/array_view.h"
#include "api/rtp_parameters.h"
#include "api/video_codecs/scalability_mode.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// SDP specification for a single video codec.
// NOTE: This class is still under development and may change without notice.
struct RTC_EXPORT SdpVideoFormat {
  using Parameters [[deprecated("Use webrtc::CodecParameterMap")]] =
      std::map<std::string, std::string>;

  explicit SdpVideoFormat(const std::string& name);
  SdpVideoFormat(const std::string& name, const CodecParameterMap& parameters);
  SdpVideoFormat(
      const std::string& name,
      const CodecParameterMap& parameters,
      const absl::InlinedVector<ScalabilityMode, kScalabilityModeCount>&
          scalability_modes);
  // Creates a new SdpVideoFormat object identical to the supplied
  // SdpVideoFormat except the scalability_modes that are set to be the same as
  // the supplied scalability modes.
  SdpVideoFormat(
      const SdpVideoFormat& format,
      const absl::InlinedVector<ScalabilityMode, kScalabilityModeCount>&
          scalability_modes);

  SdpVideoFormat(const SdpVideoFormat&);
  SdpVideoFormat(SdpVideoFormat&&);
  SdpVideoFormat& operator=(const SdpVideoFormat&);
  SdpVideoFormat& operator=(SdpVideoFormat&&);

  ~SdpVideoFormat();

  // Returns true if the SdpVideoFormats have the same names as well as codec
  // specific parameters. Please note that two SdpVideoFormats can represent the
  // same codec even though not all parameters are the same.
  bool IsSameCodec(const SdpVideoFormat& other) const;
  bool IsCodecInList(
      rtc::ArrayView<const webrtc::SdpVideoFormat> formats) const;

  std::string ToString() const;

  friend RTC_EXPORT bool operator==(const SdpVideoFormat& a,
                                    const SdpVideoFormat& b);
  friend RTC_EXPORT bool operator!=(const SdpVideoFormat& a,
                                    const SdpVideoFormat& b) {
    return !(a == b);
  }

  std::string name;
  CodecParameterMap parameters;
  absl::InlinedVector<ScalabilityMode, kScalabilityModeCount> scalability_modes;

  // Well-known video codecs and their format parameters.
  static const SdpVideoFormat VP8();
  static const SdpVideoFormat H264();
  static const SdpVideoFormat VP9Profile0();
  static const SdpVideoFormat VP9Profile1();
  static const SdpVideoFormat VP9Profile2();
  static const SdpVideoFormat VP9Profile3();
  static const SdpVideoFormat AV1Profile0();
  static const SdpVideoFormat AV1Profile1();
};

// For not so good reasons sometimes additional parameters are added to an
// SdpVideoFormat, which makes instances that should compare equal to not match
// anymore. Until we stop misusing SdpVideoFormats provide this convenience
// function to perform fuzzy matching.
std::optional<SdpVideoFormat> FuzzyMatchSdpVideoFormat(
    rtc::ArrayView<const SdpVideoFormat> supported_formats,
    const SdpVideoFormat& format);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_SDP_VIDEO_FORMAT_H_
