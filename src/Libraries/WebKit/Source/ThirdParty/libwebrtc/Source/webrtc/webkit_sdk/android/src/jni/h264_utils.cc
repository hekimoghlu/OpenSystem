/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#include "api/video_codecs/h264_profile_level_id.h"
#include "sdk/android/generated_video_jni/H264Utils_jni.h"
#include "sdk/android/src/jni/video_codec_info.h"

namespace webrtc {
namespace jni {

static jboolean JNI_H264Utils_IsSameH264Profile(
    JNIEnv* env,
    const JavaParamRef<jobject>& params1,
    const JavaParamRef<jobject>& params2) {
  return H264IsSameProfile(JavaToNativeStringMap(env, params1),
                           JavaToNativeStringMap(env, params2));
}

}  // namespace jni
}  // namespace webrtc
