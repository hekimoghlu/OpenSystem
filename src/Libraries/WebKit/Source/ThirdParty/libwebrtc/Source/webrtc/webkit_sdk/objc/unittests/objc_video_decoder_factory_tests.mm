/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#import <Foundation/Foundation.h>
#import <OCMock/OCMock.h>

#include "sdk/objc/native/src/objc_video_decoder_factory.h"

#import "base/RTCVideoDecoder.h"
#import "base/RTCVideoDecoderFactory.h"
#include "media/base/codec.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/include/video_error_codes.h"
#include "rtc_base/gunit.h"

id<RTCVideoDecoderFactory> CreateDecoderFactoryReturning(int return_code) {
  id decoderMock = OCMProtocolMock(@protocol(RTCVideoDecoder));
  OCMStub([decoderMock startDecodeWithNumberOfCores:1]).andReturn(return_code);
  OCMStub([decoderMock decode:[OCMArg any]
                    missingFrames:NO
                codecSpecificInfo:[OCMArg any]
                     renderTimeMs:0])
      .andReturn(return_code);
  OCMStub([decoderMock releaseDecoder]).andReturn(return_code);

  id decoderFactoryMock = OCMProtocolMock(@protocol(RTCVideoDecoderFactory));
  RTCVideoCodecInfo *supported = [[RTCVideoCodecInfo alloc] initWithName:@"H264" parameters:nil];
  OCMStub([decoderFactoryMock supportedCodecs]).andReturn(@[ supported ]);
  OCMStub([decoderFactoryMock createDecoder:[OCMArg any]]).andReturn(decoderMock);
  return decoderFactoryMock;
}

id<RTCVideoDecoderFactory> CreateOKDecoderFactory() {
  return CreateDecoderFactoryReturning(WEBRTC_VIDEO_CODEC_OK);
}

id<RTCVideoDecoderFactory> CreateErrorDecoderFactory() {
  return CreateDecoderFactoryReturning(WEBRTC_VIDEO_CODEC_ERROR);
}

std::unique_ptr<webrtc::VideoDecoder> GetObjCDecoder(id<RTCVideoDecoderFactory> factory) {
  webrtc::ObjCVideoDecoderFactory decoder_factory(factory);
  return decoder_factory.CreateVideoDecoder(webrtc::SdpVideoFormat(cricket::kH264CodecName));
}

#pragma mark -

TEST(ObjCVideoDecoderFactoryTest, InitDecodeReturnsOKOnSuccess) {
  std::unique_ptr<webrtc::VideoDecoder> decoder = GetObjCDecoder(CreateOKDecoderFactory());

  auto* settings = new webrtc::VideoCodec();
  EXPECT_EQ(decoder->InitDecode(settings, 1), WEBRTC_VIDEO_CODEC_OK);
}

TEST(ObjCVideoDecoderFactoryTest, InitDecodeReturnsErrorOnFail) {
  std::unique_ptr<webrtc::VideoDecoder> decoder = GetObjCDecoder(CreateErrorDecoderFactory());

  auto* settings = new webrtc::VideoCodec();
  EXPECT_EQ(decoder->InitDecode(settings, 1), WEBRTC_VIDEO_CODEC_ERROR);
}

TEST(ObjCVideoDecoderFactoryTest, DecodeReturnsOKOnSuccess) {
  std::unique_ptr<webrtc::VideoDecoder> decoder = GetObjCDecoder(CreateOKDecoderFactory());

  webrtc::EncodedImage encoded_image;
  encoded_image.SetEncodedData(webrtc::EncodedImageBuffer::Create());

  EXPECT_EQ(decoder->Decode(encoded_image, false, 0), WEBRTC_VIDEO_CODEC_OK);
}

TEST(ObjCVideoDecoderFactoryTest, DecodeReturnsErrorOnFail) {
  std::unique_ptr<webrtc::VideoDecoder> decoder = GetObjCDecoder(CreateErrorDecoderFactory());

  webrtc::EncodedImage encoded_image;
  encoded_image.SetEncodedData(webrtc::EncodedImageBuffer::Create());

  EXPECT_EQ(decoder->Decode(encoded_image, false, 0), WEBRTC_VIDEO_CODEC_ERROR);
}

TEST(ObjCVideoDecoderFactoryTest, ReleaseDecodeReturnsOKOnSuccess) {
  std::unique_ptr<webrtc::VideoDecoder> decoder = GetObjCDecoder(CreateOKDecoderFactory());

  EXPECT_EQ(decoder->Release(), WEBRTC_VIDEO_CODEC_OK);
}

TEST(ObjCVideoDecoderFactoryTest, ReleaseDecodeReturnsErrorOnFail) {
  std::unique_ptr<webrtc::VideoDecoder> decoder = GetObjCDecoder(CreateErrorDecoderFactory());

  EXPECT_EQ(decoder->Release(), WEBRTC_VIDEO_CODEC_ERROR);
}
