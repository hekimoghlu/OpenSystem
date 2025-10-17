/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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

#include "modules/video_coding/codecs/vp9/include/vp9.h"
#include "sdk/android/generated_libvpx_vp9_jni/LibvpxVp9Decoder_jni.h"
#include "sdk/android/generated_libvpx_vp9_jni/LibvpxVp9Encoder_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

static jlong JNI_LibvpxVp9Encoder_CreateEncoder(JNIEnv* jni) {
  return jlongFromPointer(VP9Encoder::Create().release());
}

static jboolean JNI_LibvpxVp9Encoder_IsSupported(JNIEnv* jni) {
  return !SupportedVP9Codecs().empty();
}

static jlong JNI_LibvpxVp9Decoder_CreateDecoder(JNIEnv* jni) {
  return jlongFromPointer(VP9Decoder::Create().release());
}

static jboolean JNI_LibvpxVp9Decoder_IsSupported(JNIEnv* jni) {
  return !SupportedVP9Codecs().empty();
}

}  // namespace jni
}  // namespace webrtc
