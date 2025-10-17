/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#include "webkit_sdk/objc/native/src/objc_video_decoder_factory.h"

#import "base/RTCVideoDecoder.h"
#import "base/RTCVideoDecoderFactory.h"
#import "base/RTCVideoFrame.h"
#import "base/RTCVideoFrameBuffer.h"
#import "components/video_codec/RTCCodecSpecificInfoH264.h"
#import "webkit_sdk/objc/api/peerconnection/RTCEncodedImage+Private.h"
#import "webkit_sdk/objc/api/peerconnection/RTCVideoCodecInfo+Private.h"
#import "webkit_sdk/objc/api/video_codec/RTCWrappedNativeVideoDecoder.h"
#import "webkit_sdk/objc/helpers/NSString+StdString.h"

#include "api/make_ref_counted.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/include/video_error_codes.h"
#include "rtc_base/logging.h"
#include "rtc_base/time_utils.h"
#include "webkit_sdk/objc/native/src/objc_frame_buffer.h"

namespace webrtc {

namespace {
class ObjCVideoDecoder : public VideoDecoder {
 public:
  ObjCVideoDecoder(id<RTCVideoDecoder> decoder)
      : decoder_(decoder), implementation_name_([decoder implementationName].rtcStdString) {}

  bool Configure(const Settings& settings) override {
    return [decoder_ startDecodeWithNumberOfCores:settings.number_of_cores()] == WEBRTC_VIDEO_CODEC_OK;
  }

  int32_t Decode(const EncodedImage &input_image,
                 bool missing_frames,
                 int64_t render_time_ms = -1) override {
    RTCEncodedImage *encodedImage =
        [[RTCEncodedImage alloc] initWithNativeEncodedImage:input_image];

    return [decoder_ decode:encodedImage
              missingFrames:missing_frames
          codecSpecificInfo:nil
               renderTimeMs:render_time_ms];
  }

  int32_t RegisterDecodeCompleteCallback(DecodedImageCallback *callback) override {
    [decoder_ setCallback:^(RTCVideoFrame *frame, bool) {
      const rtc::scoped_refptr<VideoFrameBuffer> buffer =
          rtc::make_ref_counted<ObjCFrameBuffer>(frame.buffer);
      VideoFrame videoFrame =
          VideoFrame::Builder()
              .set_video_frame_buffer(buffer)
              .set_timestamp_rtp(frame.timeStamp)
              .set_timestamp_ms(0)
              .set_rotation((VideoRotation)frame.rotation)
              .build();

      callback->Decoded(videoFrame);
    }];

    return WEBRTC_VIDEO_CODEC_OK;
  }

  int32_t Release() override { return [decoder_ releaseDecoder]; }

  const char *ImplementationName() const override { return implementation_name_.c_str(); }

 private:
  id<RTCVideoDecoder> decoder_;
  const std::string implementation_name_;
};
}  // namespace

ObjCVideoDecoderFactory::ObjCVideoDecoderFactory(id<RTCVideoDecoderFactory> decoder_factory)
    : decoder_factory_(decoder_factory) {}

ObjCVideoDecoderFactory::~ObjCVideoDecoderFactory() {}

id<RTCVideoDecoderFactory> ObjCVideoDecoderFactory::wrapped_decoder_factory() const {
  return decoder_factory_;
}

ObjCVideoDecoderFactory::CodecSupport ObjCVideoDecoderFactory::QueryCodecSupport(const SdpVideoFormat& format, bool reference_scaling) const {
    CodecSupport codec_support;
    codec_support.is_supported = format.IsCodecInList(GetSupportedFormats());
    return codec_support;
}

std::unique_ptr<VideoDecoder> ObjCVideoDecoderFactory::Create(
    const Environment& environment,
    const SdpVideoFormat &format) {
  NSString *codecName = [NSString stringWithUTF8String:format.name.c_str()];
  for (RTCVideoCodecInfo *codecInfo in decoder_factory_.supportedCodecs) {
    if ([codecName isEqualToString:codecInfo.name]) {
      id<RTCVideoDecoder> decoder = [decoder_factory_ createDecoder:codecInfo];

      // Because of symbol conflict, isKindOfClass doesn't work as expected.
      // See https://bugs.webkit.org/show_bug.cgi?id=198782.
      // if ([decoder isKindOfClass:[RTCWrappedNativeVideoDecoder class]]) {
#if !defined(WEBRTC_WEBKIT_BUILD)
        if ([codecName isEqual:@"VP8"] || [codecName isEqual:@"VP9"]) {
#else
        if (![decoder.implementationName isEqual:@"VideoToolbox"]) {
#endif
        return [(RTCWrappedNativeVideoDecoder *)decoder releaseWrappedDecoder];
      } else {
        return std::unique_ptr<ObjCVideoDecoder>(new ObjCVideoDecoder(decoder));
      }
    }
  }

  return nullptr;
}

std::vector<SdpVideoFormat> ObjCVideoDecoderFactory::GetSupportedFormats() const {
  std::vector<SdpVideoFormat> supported_formats;
  for (RTCVideoCodecInfo *supportedCodec in decoder_factory_.supportedCodecs) {
    SdpVideoFormat format = [supportedCodec nativeSdpVideoFormat];
    supported_formats.push_back(format);
  }

  return supported_formats;
}

}  // namespace webrtc
