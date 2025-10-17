/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#include "test/scenario/hardware_codecs.h"

#include "rtc_base/checks.h"

#ifdef WEBRTC_ANDROID
#include "modules/video_coding/codecs/test/android_codec_factory_helper.h"
#endif
#ifdef WEBRTC_MAC
#include "modules/video_coding/codecs/test/objc_codec_factory_helper.h"
#endif

namespace webrtc {
namespace test {
std::unique_ptr<VideoEncoderFactory> CreateHardwareEncoderFactory() {
#ifdef WEBRTC_ANDROID
  InitializeAndroidObjects();
  return CreateAndroidEncoderFactory();
#else
#ifdef WEBRTC_MAC
  return CreateObjCEncoderFactory();
#else
  RTC_DCHECK_NOTREACHED()
      << "Hardware encoder not implemented on this platform.";
  return nullptr;
#endif
#endif
}
std::unique_ptr<VideoDecoderFactory> CreateHardwareDecoderFactory() {
#ifdef WEBRTC_ANDROID
  InitializeAndroidObjects();
  return CreateAndroidDecoderFactory();
#else
#ifdef WEBRTC_MAC
  return CreateObjCDecoderFactory();
#else
  RTC_DCHECK_NOTREACHED()
      << "Hardware decoder not implemented on this platform.";
  return nullptr;
#endif
#endif
}
}  // namespace test
}  // namespace webrtc
