/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"

#include <cstdint>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtp_parameters.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/corruption_detection_extension.h"
#include "modules/rtp_rtcp/source/rtp_dependency_descriptor_extension.h"
#include "modules/rtp_rtcp/source/rtp_generic_frame_descriptor_extension.h"
#include "modules/rtp_rtcp/source/rtp_header_extensions.h"
#include "modules/rtp_rtcp/source/rtp_video_layers_allocation_extension.h"
#include "rtc_base/arraysize.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {

struct ExtensionInfo {
  RTPExtensionType type;
  absl::string_view uri;
};

template <typename Extension>
constexpr ExtensionInfo CreateExtensionInfo() {
  return {Extension::kId, Extension::Uri()};
}

constexpr ExtensionInfo kExtensions[] = {
    CreateExtensionInfo<TransmissionOffset>(),
    CreateExtensionInfo<AudioLevelExtension>(),
    CreateExtensionInfo<CsrcAudioLevel>(),
    CreateExtensionInfo<AbsoluteSendTime>(),
    CreateExtensionInfo<AbsoluteCaptureTimeExtension>(),
    CreateExtensionInfo<VideoOrientation>(),
    CreateExtensionInfo<TransportSequenceNumber>(),
    CreateExtensionInfo<TransportSequenceNumberV2>(),
    CreateExtensionInfo<PlayoutDelayLimits>(),
    CreateExtensionInfo<VideoContentTypeExtension>(),
    CreateExtensionInfo<RtpVideoLayersAllocationExtension>(),
    CreateExtensionInfo<VideoTimingExtension>(),
    CreateExtensionInfo<RtpStreamId>(),
    CreateExtensionInfo<RepairedRtpStreamId>(),
    CreateExtensionInfo<RtpMid>(),
    CreateExtensionInfo<RtpGenericFrameDescriptorExtension00>(),
    CreateExtensionInfo<RtpDependencyDescriptorExtension>(),
    CreateExtensionInfo<ColorSpaceExtension>(),
    CreateExtensionInfo<InbandComfortNoiseExtension>(),
    CreateExtensionInfo<VideoFrameTrackingIdExtension>(),
    CreateExtensionInfo<CorruptionDetectionExtension>(),
};

// Because of kRtpExtensionNone, NumberOfExtension is 1 bigger than the actual
// number of known extensions.
static_assert(arraysize(kExtensions) ==
                  static_cast<int>(kRtpExtensionNumberOfExtensions) - 1,
              "kExtensions expect to list all known extensions");

}  // namespace

RtpHeaderExtensionMap::RtpHeaderExtensionMap() : RtpHeaderExtensionMap(false) {}

RtpHeaderExtensionMap::RtpHeaderExtensionMap(bool extmap_allow_mixed)
    : extmap_allow_mixed_(extmap_allow_mixed) {
  for (auto& id : ids_)
    id = kInvalidId;
}

RtpHeaderExtensionMap::RtpHeaderExtensionMap(
    rtc::ArrayView<const RtpExtension> extensions)
    : RtpHeaderExtensionMap(false) {
  for (const RtpExtension& extension : extensions)
    RegisterByUri(extension.id, extension.uri);
}

void RtpHeaderExtensionMap::Reset(
    rtc::ArrayView<const RtpExtension> extensions) {
  for (auto& id : ids_)
    id = kInvalidId;
  for (const RtpExtension& extension : extensions)
    RegisterByUri(extension.id, extension.uri);
}

bool RtpHeaderExtensionMap::RegisterByType(int id, RTPExtensionType type) {
  for (const ExtensionInfo& extension : kExtensions)
    if (type == extension.type)
      return Register(id, extension.type, extension.uri);
  RTC_DCHECK_NOTREACHED();
  return false;
}

bool RtpHeaderExtensionMap::RegisterByUri(int id, absl::string_view uri) {
  for (const ExtensionInfo& extension : kExtensions)
    if (uri == extension.uri)
      return Register(id, extension.type, extension.uri);
  RTC_LOG(LS_WARNING) << "Unknown extension uri:'" << uri << "', id: " << id
                      << '.';
  return false;
}

RTPExtensionType RtpHeaderExtensionMap::GetType(int id) const {
  RTC_DCHECK_GE(id, RtpExtension::kMinId);
  RTC_DCHECK_LE(id, RtpExtension::kMaxId);
  for (int type = kRtpExtensionNone + 1; type < kRtpExtensionNumberOfExtensions;
       ++type) {
    if (ids_[type] == id) {
      return static_cast<RTPExtensionType>(type);
    }
  }
  return kInvalidType;
}

void RtpHeaderExtensionMap::Deregister(absl::string_view uri) {
  for (const ExtensionInfo& extension : kExtensions) {
    if (extension.uri == uri) {
      ids_[extension.type] = kInvalidId;
      break;
    }
  }
}

bool RtpHeaderExtensionMap::Register(int id,
                                     RTPExtensionType type,
                                     absl::string_view uri) {
  RTC_DCHECK_GT(type, kRtpExtensionNone);
  RTC_DCHECK_LT(type, kRtpExtensionNumberOfExtensions);

  if (id < RtpExtension::kMinId || id > RtpExtension::kMaxId) {
    RTC_LOG(LS_WARNING) << "Failed to register extension uri:'" << uri
                        << "' with invalid id:" << id << ".";
    return false;
  }

  RTPExtensionType registered_type = GetType(id);
  if (registered_type == type) {  // Same type/id pair already registered.
    RTC_LOG(LS_VERBOSE) << "Reregistering extension uri:'" << uri
                        << "', id:" << id;
    return true;
  }

  if (registered_type !=
      kInvalidType) {  // `id` used by another extension type.
    RTC_LOG(LS_WARNING) << "Failed to register extension uri:'" << uri
                        << "', id:" << id
                        << ". Id already in use by extension type "
                        << static_cast<int>(registered_type);
    return false;
  }
  if (IsRegistered(type)) {
    RTC_LOG(LS_WARNING) << "Illegal reregistration for uri: " << uri
                        << " is previously registered with id " << GetId(type)
                        << " and cannot be reregistered with id " << id;
    return false;
  }

  // There is a run-time check above id fits into uint8_t.
  ids_[type] = static_cast<uint8_t>(id);
  return true;
}

}  // namespace webrtc
