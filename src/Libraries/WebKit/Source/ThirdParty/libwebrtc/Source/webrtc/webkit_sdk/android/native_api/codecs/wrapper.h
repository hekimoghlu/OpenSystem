/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#ifndef SDK_ANDROID_NATIVE_API_CODECS_WRAPPER_H_
#define SDK_ANDROID_NATIVE_API_CODECS_WRAPPER_H_

#include <jni.h>

#include <memory>
#include <vector>

#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/video_encoder_factory.h"

namespace webrtc {

// Creates an instance of webrtc::SdpVideoFormat from Java VideoCodecInfo.
SdpVideoFormat JavaToNativeVideoCodecInfo(JNIEnv* jni, jobject codec_info);

// Creates an instance of webrtc::VideoDecoderFactory from Java
// VideoDecoderFactory.
std::unique_ptr<VideoDecoderFactory> JavaToNativeVideoDecoderFactory(
    JNIEnv* jni,
    jobject decoder_factory);

// Creates an instance of webrtc::VideoEncoderFactory from Java
// VideoEncoderFactory.
std::unique_ptr<VideoEncoderFactory> JavaToNativeVideoEncoderFactory(
    JNIEnv* jni,
    jobject encoder_factory);

// Creates an array of VideoEncoder::ResolutionBitrateLimits from Java array
// of ResolutionBitrateLimits.
std::vector<VideoEncoder::ResolutionBitrateLimits>
JavaToNativeResolutionBitrateLimits(JNIEnv* jni,
                                    jobjectArray j_bitrate_limits_array);

}  // namespace webrtc

#endif  // SDK_ANDROID_NATIVE_API_CODECS_WRAPPER_H_
