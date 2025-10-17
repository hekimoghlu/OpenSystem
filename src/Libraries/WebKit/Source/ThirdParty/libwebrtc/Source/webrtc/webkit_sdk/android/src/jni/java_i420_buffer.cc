/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "sdk/android/generated_video_jni/JavaI420Buffer_jni.h"
#include "third_party/libyuv/include/libyuv/scale.h"

namespace webrtc {
namespace jni {

static void JNI_JavaI420Buffer_CropAndScaleI420(
    JNIEnv* jni,
    const JavaParamRef<jobject>& j_src_y,
    jint src_stride_y,
    const JavaParamRef<jobject>& j_src_u,
    jint src_stride_u,
    const JavaParamRef<jobject>& j_src_v,
    jint src_stride_v,
    jint crop_x,
    jint crop_y,
    jint crop_width,
    jint crop_height,
    const JavaParamRef<jobject>& j_dst_y,
    jint dst_stride_y,
    const JavaParamRef<jobject>& j_dst_u,
    jint dst_stride_u,
    const JavaParamRef<jobject>& j_dst_v,
    jint dst_stride_v,
    jint scale_width,
    jint scale_height) {
  uint8_t const* src_y =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_src_y.obj()));
  uint8_t const* src_u =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_src_u.obj()));
  uint8_t const* src_v =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_src_v.obj()));
  uint8_t* dst_y =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_y.obj()));
  uint8_t* dst_u =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_u.obj()));
  uint8_t* dst_v =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_v.obj()));

  // Perform cropping using pointer arithmetic.
  src_y += crop_x + crop_y * src_stride_y;
  src_u += crop_x / 2 + crop_y / 2 * src_stride_u;
  src_v += crop_x / 2 + crop_y / 2 * src_stride_v;

  bool ret = libyuv::I420Scale(
      src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v, crop_width,
      crop_height, dst_y, dst_stride_y, dst_u, dst_stride_u, dst_v,
      dst_stride_v, scale_width, scale_height, libyuv::kFilterBox);
  RTC_DCHECK_EQ(ret, 0) << "I420Scale failed";
}

}  // namespace jni
}  // namespace webrtc
