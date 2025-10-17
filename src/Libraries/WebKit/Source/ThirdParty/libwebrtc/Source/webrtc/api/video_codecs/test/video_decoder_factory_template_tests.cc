/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/test/mock_video_decoder.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_decoder_factory_template.h"
#include "api/video_codecs/video_decoder_factory_template_dav1d_adapter.h"
#include "api/video_codecs/video_decoder_factory_template_libvpx_vp8_adapter.h"
#include "api/video_codecs/video_decoder_factory_template_libvpx_vp9_adapter.h"
#include "api/video_codecs/video_decoder_factory_template_open_h264_adapter.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::Each;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::UnorderedElementsAre;

const SdpVideoFormat kFooSdp("Foo");
const SdpVideoFormat kBarLowSdp("Bar", {{"profile", "low"}});
const SdpVideoFormat kBarHighSdp("Bar", {{"profile", "high"}});

struct FooDecoderTemplateAdapter {
  static std::vector<SdpVideoFormat> SupportedFormats() { return {kFooSdp}; }

  static std::unique_ptr<VideoDecoder> CreateDecoder(
      const SdpVideoFormat& /* format */) {
    auto decoder = std::make_unique<testing::StrictMock<MockVideoDecoder>>();
    EXPECT_CALL(*decoder, Destruct);
    return decoder;
  }
};

struct BarDecoderTemplateAdapter {
  static std::vector<SdpVideoFormat> SupportedFormats() {
    return {kBarLowSdp, kBarHighSdp};
  }

  static std::unique_ptr<VideoDecoder> CreateDecoder(
      const Environment& /* env */,
      const SdpVideoFormat& /* format */) {
    auto decoder = std::make_unique<testing::StrictMock<MockVideoDecoder>>();
    EXPECT_CALL(*decoder, Destruct);
    return decoder;
  }
};

TEST(VideoDecoderFactoryTemplate, OneTemplateAdapterCreateDecoder) {
  const Environment env = CreateEnvironment();
  VideoDecoderFactoryTemplate<FooDecoderTemplateAdapter> factory;
  EXPECT_THAT(factory.GetSupportedFormats(), UnorderedElementsAre(kFooSdp));
  EXPECT_THAT(factory.Create(env, kFooSdp), NotNull());
  EXPECT_THAT(factory.Create(env, SdpVideoFormat("FooX")), IsNull());
}

TEST(VideoDecoderFactoryTemplate, TwoTemplateAdaptersNoDuplicates) {
  VideoDecoderFactoryTemplate<FooDecoderTemplateAdapter,
                              FooDecoderTemplateAdapter>
      factory;
  EXPECT_THAT(factory.GetSupportedFormats(), UnorderedElementsAre(kFooSdp));
}

TEST(VideoDecoderFactoryTemplate, TwoTemplateAdaptersCreateDecoders) {
  const Environment env = CreateEnvironment();
  VideoDecoderFactoryTemplate<FooDecoderTemplateAdapter,
                              BarDecoderTemplateAdapter>
      factory;
  EXPECT_THAT(factory.GetSupportedFormats(),
              UnorderedElementsAre(kFooSdp, kBarLowSdp, kBarHighSdp));
  EXPECT_THAT(factory.Create(env, kFooSdp), NotNull());
  EXPECT_THAT(factory.Create(env, kBarLowSdp), NotNull());
  EXPECT_THAT(factory.Create(env, kBarHighSdp), NotNull());
  EXPECT_THAT(factory.Create(env, SdpVideoFormat("FooX")), IsNull());
  EXPECT_THAT(factory.Create(env, SdpVideoFormat("Bar")), IsNull());
}

TEST(VideoDecoderFactoryTemplate, LibvpxVp8) {
  const Environment env = CreateEnvironment();
  VideoDecoderFactoryTemplate<LibvpxVp8DecoderTemplateAdapter> factory;
  auto formats = factory.GetSupportedFormats();
  ASSERT_THAT(formats,
              UnorderedElementsAre(Field(&SdpVideoFormat::name, "VP8")));
  EXPECT_THAT(factory.Create(env, formats[0]), NotNull());
}

TEST(VideoDecoderFactoryTemplate, LibvpxVp9) {
  const Environment env = CreateEnvironment();
  VideoDecoderFactoryTemplate<LibvpxVp9DecoderTemplateAdapter> factory;
  auto formats = factory.GetSupportedFormats();
  EXPECT_THAT(formats, Not(IsEmpty()));
  EXPECT_THAT(formats, Each(Field(&SdpVideoFormat::name, "VP9")));
  EXPECT_THAT(factory.Create(env, formats[0]), NotNull());
}

// TODO(bugs.webrtc.org/13573): When OpenH264 is no longer a conditional build
//                              target remove this #ifdef.
#if defined(WEBRTC_USE_H264)
TEST(VideoDecoderFactoryTemplate, OpenH264) {
  const Environment env = CreateEnvironment();
  VideoDecoderFactoryTemplate<OpenH264DecoderTemplateAdapter> factory;
  auto formats = factory.GetSupportedFormats();
  EXPECT_THAT(formats, Not(IsEmpty()));
  EXPECT_THAT(formats, Each(Field(&SdpVideoFormat::name, "H264")));
  EXPECT_THAT(factory.Create(env, formats[0]), NotNull());
}
#endif  // defined(WEBRTC_USE_H264)

TEST(VideoDecoderFactoryTemplate, Dav1d) {
  const Environment env = CreateEnvironment();
  VideoDecoderFactoryTemplate<Dav1dDecoderTemplateAdapter> factory;
  auto formats = factory.GetSupportedFormats();
  EXPECT_THAT(formats, Not(IsEmpty()));
  EXPECT_THAT(formats, Each(Field(&SdpVideoFormat::name, "AV1")));
  EXPECT_THAT(factory.Create(env, formats[0]), NotNull());
}

}  // namespace
}  // namespace webrtc
