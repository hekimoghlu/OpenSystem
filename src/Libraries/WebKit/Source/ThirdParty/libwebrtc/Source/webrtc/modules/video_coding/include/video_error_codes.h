/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#ifndef MODULES_VIDEO_CODING_INCLUDE_VIDEO_ERROR_CODES_H_
#define MODULES_VIDEO_CODING_INCLUDE_VIDEO_ERROR_CODES_H_

#define WEBRTC_VIDEO_CODEC_TARGET_BITRATE_OVERSHOOT 5
#define WEBRTC_VIDEO_CODEC_OK_REQUEST_KEYFRAME 4
#define WEBRTC_VIDEO_CODEC_NO_OUTPUT 1
#define WEBRTC_VIDEO_CODEC_OK 0
#define WEBRTC_VIDEO_CODEC_ERROR -1
#define WEBRTC_VIDEO_CODEC_MEMORY -3
#define WEBRTC_VIDEO_CODEC_ERR_PARAMETER -4
#define WEBRTC_VIDEO_CODEC_TIMEOUT -6
#define WEBRTC_VIDEO_CODEC_UNINITIALIZED -7
#define WEBRTC_VIDEO_CODEC_FALLBACK_SOFTWARE -13
#define WEBRTC_VIDEO_CODEC_ERR_SIMULCAST_PARAMETERS_NOT_SUPPORTED -15
#define WEBRTC_VIDEO_CODEC_ENCODER_FAILURE -16

#endif  // MODULES_VIDEO_CODING_INCLUDE_VIDEO_ERROR_CODES_H_
