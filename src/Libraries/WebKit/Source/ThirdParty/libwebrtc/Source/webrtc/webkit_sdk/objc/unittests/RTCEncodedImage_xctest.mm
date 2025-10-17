/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#import "api/peerconnection/RTCEncodedImage+Private.h"

#import <XCTest/XCTest.h>

@interface RTCEncodedImageTests : XCTestCase
@end

@implementation RTCEncodedImageTests

- (void)testInitializedWithNativeEncodedImage {
  const auto encoded_data = webrtc::EncodedImageBuffer::Create();
  webrtc::EncodedImage encoded_image;
  encoded_image.SetEncodedData(encoded_data);

  RTCEncodedImage *encodedImage =
      [[RTCEncodedImage alloc] initWithNativeEncodedImage:encoded_image];

  XCTAssertEqual([encodedImage nativeEncodedImage].GetEncodedData(), encoded_data);
}

- (void)testInitWithNSData {
  NSData *bufferData = [NSData data];
  RTCEncodedImage *encodedImage = [[RTCEncodedImage alloc] init];
  encodedImage.buffer = bufferData;

  webrtc::EncodedImage result_encoded_image = [encodedImage nativeEncodedImage];
  XCTAssertTrue(result_encoded_image.GetEncodedData() != nullptr);
  XCTAssertEqual(result_encoded_image.GetEncodedData()->data(), bufferData.bytes);
}

- (void)testRetainsNativeEncodedImage {
  RTCEncodedImage *encodedImage;
  {
    const auto encoded_data = webrtc::EncodedImageBuffer::Create();
    webrtc::EncodedImage encoded_image;
    encoded_image.SetEncodedData(encoded_data);
    encodedImage = [[RTCEncodedImage alloc] initWithNativeEncodedImage:encoded_image];
  }
  webrtc::EncodedImage result_encoded_image = [encodedImage nativeEncodedImage];
  XCTAssertTrue(result_encoded_image.GetEncodedData() != nullptr);
  XCTAssertTrue(result_encoded_image.GetEncodedData()->data() != nullptr);
}

@end
