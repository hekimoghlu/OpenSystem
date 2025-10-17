/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#import "RTCEncodedImage.h"

@implementation RTCEncodedImage

@synthesize buffer = _buffer;
@synthesize encodedWidth = _encodedWidth;
@synthesize encodedHeight = _encodedHeight;
@synthesize timeStamp = _timeStamp;
@synthesize duration = _duration;
@synthesize captureTimeMs = _captureTimeMs;
@synthesize ntpTimeMs = _ntpTimeMs;
@synthesize flags = _flags;
@synthesize encodeStartMs = _encodeStartMs;
@synthesize encodeFinishMs = _encodeFinishMs;
@synthesize frameType = _frameType;
@synthesize rotation = _rotation;
@synthesize qp = _qp;
@synthesize contentType = _contentType;

@end
