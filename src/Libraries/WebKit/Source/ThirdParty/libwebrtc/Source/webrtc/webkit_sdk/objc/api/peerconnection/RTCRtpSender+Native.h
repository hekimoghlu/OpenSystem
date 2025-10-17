/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
#import "RTCRtpSender.h"

#include "api/crypto/frame_encryptor_interface.h"
#include "api/scoped_refptr.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * This class extension exposes methods that work directly with injectable C++ components.
 */
@interface RTCRtpSender ()

/** Sets a defined frame encryptor that will encrypt the entire frame
 * before it is sent across the network. This will encrypt the entire frame
 * using the user provided encryption mechanism regardless of whether SRTP is
 * enabled or not.
 */
- (void)setFrameEncryptor:(rtc::scoped_refptr<webrtc::FrameEncryptorInterface>)frameEncryptor;

@end

NS_ASSUME_NONNULL_END
