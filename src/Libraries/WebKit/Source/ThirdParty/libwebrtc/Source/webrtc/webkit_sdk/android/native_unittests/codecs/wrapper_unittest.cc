/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "sdk/android/native_api/codecs/wrapper.h"

#include <memory>

#include "absl/memory/memory.h"
#include "media/base/media_constants.h"
#include "sdk/android/generated_native_unittests_jni/CodecsWrapperTestHelper_jni.h"
#include "sdk/android/src/jni/video_encoder_wrapper.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {
namespace {
TEST(JavaCodecsWrapperTest, JavaToNativeVideoCodecInfo) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jobject> j_video_codec_info =
      jni::Java_CodecsWrapperTestHelper_createTestVideoCodecInfo(env);

  const SdpVideoFormat video_format =
      JavaToNativeVideoCodecInfo(env, j_video_codec_info.obj());

  EXPECT_EQ(cricket::kH264CodecName, video_format.name);
  const auto it =
      video_format.parameters.find(cricket::kH264FmtpProfileLevelId);
  ASSERT_NE(it, video_format.parameters.end());
  EXPECT_EQ(cricket::kH264ProfileLevelConstrainedBaseline, it->second);
}

TEST(JavaCodecsWrapperTest, JavaToNativeResolutionBitrateLimits) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jobject> j_fake_encoder =
      jni::Java_CodecsWrapperTestHelper_createFakeVideoEncoder(env);

  auto encoder = jni::JavaToNativeVideoEncoder(env, j_fake_encoder);
  ASSERT_TRUE(encoder);

  // Check that the bitrate limits correctly passed from Java to native.
  const std::vector<VideoEncoder::ResolutionBitrateLimits> bitrate_limits =
      encoder->GetEncoderInfo().resolution_bitrate_limits;
  ASSERT_EQ(bitrate_limits.size(), 1u);
  EXPECT_EQ(bitrate_limits[0].frame_size_pixels, 640 * 360);
  EXPECT_EQ(bitrate_limits[0].min_start_bitrate_bps, 300000);
  EXPECT_EQ(bitrate_limits[0].min_bitrate_bps, 200000);
  EXPECT_EQ(bitrate_limits[0].max_bitrate_bps, 1000000);
}
}  // namespace
}  // namespace test
}  // namespace webrtc
