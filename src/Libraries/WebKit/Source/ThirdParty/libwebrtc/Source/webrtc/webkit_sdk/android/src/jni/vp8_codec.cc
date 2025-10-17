/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#include <jni.h>

#include "modules/video_coding/codecs/vp8/include/vp8.h"
#include "sdk/android/generated_libvpx_vp8_jni/LibvpxVp8Decoder_jni.h"
#include "sdk/android/generated_libvpx_vp8_jni/LibvpxVp8Encoder_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

static jlong JNI_LibvpxVp8Encoder_CreateEncoder(JNIEnv* jni) {
  return jlongFromPointer(VP8Encoder::Create().release());
}

static jlong JNI_LibvpxVp8Decoder_CreateDecoder(JNIEnv* jni) {
  return jlongFromPointer(VP8Decoder::Create().release());
}

}  // namespace jni
}  // namespace webrtc
