/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#ifndef MODULES_VIDEO_CODING_DEPRECATED_JITTER_BUFFER_COMMON_H_
#define MODULES_VIDEO_CODING_DEPRECATED_JITTER_BUFFER_COMMON_H_

namespace webrtc {

// Used to estimate rolling average of packets per frame.
static const float kFastConvergeMultiplier = 0.4f;
static const float kNormalConvergeMultiplier = 0.2f;

enum { kMaxNumberOfFrames = 300 };
enum { kStartNumberOfFrames = 6 };
enum { kMaxVideoDelayMs = 10000 };
enum { kPacketsPerFrameMultiplier = 5 };
enum { kFastConvergeThreshold = 5 };

enum VCMJitterBufferEnum {
  kMaxConsecutiveOldFrames = 60,
  kMaxConsecutiveOldPackets = 300,
  // TODO(sprang): Reduce this limit once codecs don't sometimes wildly
  // overshoot bitrate target.
  kMaxPacketsInSession = 1400,      // Allows ~2MB frames.
  kBufferIncStepSizeBytes = 30000,  // >20 packets.
  kMaxJBFrameSizeBytes = 4000000    // sanity don't go above 4Mbyte.
};

enum VCMFrameBufferEnum {
  kOutOfBoundsPacket = -7,
  kNotInitialized = -6,
  kOldPacket = -5,
  kGeneralError = -4,
  kFlushIndicator = -3,  // Indicator that a flush has occurred.
  kTimeStampError = -2,
  kSizeError = -1,
  kNoError = 0,
  kIncomplete = 1,       // Frame incomplete.
  kCompleteSession = 3,  // at least one layer in the frame complete.
  kDuplicatePacket = 5   // We're receiving a duplicate packet.
};

enum VCMFrameBufferStateEnum {
  kStateEmpty,       // frame popped by the RTP receiver
  kStateIncomplete,  // frame that have one or more packet(s) stored
  kStateComplete,    // frame that have all packets
};

enum { kH264StartCodeLengthBytes = 4 };
}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_DEPRECATED_JITTER_BUFFER_COMMON_H_
