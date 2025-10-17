/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#ifdef __cplusplus
extern "C" {
#endif
#import <OCMock/OCMock.h>
#ifdef __cplusplus
}
#endif
#import "api/peerconnection/RTCPeerConnectionFactory+Native.h"
#import "api/peerconnection/RTCPeerConnectionFactoryBuilder+DefaultComponents.h"
#import "api/peerconnection/RTCPeerConnectionFactoryBuilder.h"

#include "api/audio_codecs/builtin_audio_decoder_factory.h"
#include "api/audio_codecs/builtin_audio_encoder_factory.h"
#include "api/transport/media/media_transport_interface.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "modules/audio_device/include/audio_device.h"
#include "modules/audio_processing/include/audio_processing.h"

#include "rtc_base/gunit.h"
#include "rtc_base/system/unused.h"

@interface RTCPeerConnectionFactoryBuilderTest : NSObject
- (void)testBuilder;
- (void)testDefaultComponentsBuilder;
@end

@implementation RTCPeerConnectionFactoryBuilderTest

- (void)testBuilder {
  id factoryMock = OCMStrictClassMock([RTCPeerConnectionFactory class]);
  OCMExpect([factoryMock alloc]).andReturn(factoryMock);
#ifdef HAVE_NO_MEDIA
  RTC_UNUSED([[[factoryMock expect] andReturn:factoryMock] initWithNoMedia]);
#else
  RTC_UNUSED([[[[factoryMock expect] andReturn:factoryMock] ignoringNonObjectArgs]
      initWithNativeAudioEncoderFactory:nullptr
              nativeAudioDecoderFactory:nullptr
              nativeVideoEncoderFactory:nullptr
              nativeVideoDecoderFactory:nullptr
                      audioDeviceModule:nullptr
                  audioProcessingModule:nullptr
                  mediaTransportFactory:nullptr]);
#endif
  RTCPeerConnectionFactoryBuilder* builder = [[RTCPeerConnectionFactoryBuilder alloc] init];
  RTCPeerConnectionFactory* peerConnectionFactory = [builder createPeerConnectionFactory];
  EXPECT_TRUE(peerConnectionFactory != nil);
  OCMVerifyAll(factoryMock);
}

- (void)testDefaultComponentsBuilder {
  id factoryMock = OCMStrictClassMock([RTCPeerConnectionFactory class]);
  OCMExpect([factoryMock alloc]).andReturn(factoryMock);
#ifdef HAVE_NO_MEDIA
  RTC_UNUSED([[[factoryMock expect] andReturn:factoryMock] initWithNoMedia]);
#else
  RTC_UNUSED([[[[factoryMock expect] andReturn:factoryMock] ignoringNonObjectArgs]
      initWithNativeAudioEncoderFactory:nullptr
              nativeAudioDecoderFactory:nullptr
              nativeVideoEncoderFactory:nullptr
              nativeVideoDecoderFactory:nullptr
                      audioDeviceModule:nullptr
                  audioProcessingModule:nullptr
                  mediaTransportFactory:nullptr]);
#endif
  RTCPeerConnectionFactoryBuilder* builder = [RTCPeerConnectionFactoryBuilder defaultBuilder];
  RTCPeerConnectionFactory* peerConnectionFactory = [builder createPeerConnectionFactory];
  EXPECT_TRUE(peerConnectionFactory != nil);
  OCMVerifyAll(factoryMock);
}
@end

TEST(RTCPeerConnectionFactoryBuilderTest, BuilderTest) {
  @autoreleasepool {
    RTCPeerConnectionFactoryBuilderTest* test = [[RTCPeerConnectionFactoryBuilderTest alloc] init];
    [test testBuilder];
  }
}

TEST(RTCPeerConnectionFactoryBuilderTest, DefaultComponentsBuilderTest) {
  @autoreleasepool {
    RTCPeerConnectionFactoryBuilderTest* test = [[RTCPeerConnectionFactoryBuilderTest alloc] init];
    [test testDefaultComponentsBuilder];
  }
}
