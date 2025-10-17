/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#ifndef MEDIA_ENGINE_WEBRTC_MEDIA_ENGINE_H_
#define MEDIA_ENGINE_WEBRTC_MEDIA_ENGINE_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/field_trials_view.h"
#include "api/rtp_parameters.h"
#include "api/transport/bitrate_settings.h"
#include "media/base/codec.h"

namespace cricket {

// Verify that extension IDs are within 1-byte extension range and are not
// overlapping, and that they form a legal change from previously registerd
// extensions (if any).
bool ValidateRtpExtensions(
    rtc::ArrayView<const webrtc::RtpExtension> extennsions,
    rtc::ArrayView<const webrtc::RtpExtension> old_extensions);

// Discard any extensions not validated by the 'supported' predicate. Duplicate
// extensions are removed if 'filter_redundant_extensions' is set, and also any
// mutually exclusive extensions (see implementation for details) are removed.
std::vector<webrtc::RtpExtension> FilterRtpExtensions(
    const std::vector<webrtc::RtpExtension>& extensions,
    bool (*supported)(absl::string_view),
    bool filter_redundant_extensions,
    const webrtc::FieldTrialsView& trials);

webrtc::BitrateConstraints GetBitrateConfigForCodec(const Codec& codec);

}  // namespace cricket

#endif  // MEDIA_ENGINE_WEBRTC_MEDIA_ENGINE_H_
