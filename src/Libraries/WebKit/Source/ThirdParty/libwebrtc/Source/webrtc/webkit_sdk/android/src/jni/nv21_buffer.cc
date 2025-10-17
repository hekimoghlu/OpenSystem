/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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

#include <vector>

#include "common_video/libyuv/include/webrtc_libyuv.h"
#include "rtc_base/checks.h"
#include "sdk/android/generated_video_jni/NV21Buffer_jni.h"
#include "third_party/libyuv/include/libyuv/convert.h"
#include "third_party/libyuv/include/libyuv/scale.h"

namespace webrtc {
namespace jni {

static void JNI_NV21Buffer_CropAndScale(JNIEnv* jni,
                                        jint crop_x,
                                        jint crop_y,
                                        jint crop_width,
                                        jint crop_height,
                                        jint scale_width,
                                        jint scale_height,
                                        const JavaParamRef<jbyteArray>& j_src,
                                        jint src_width,
                                        jint src_height,
                                        const JavaParamRef<jobject>& j_dst_y,
                                        jint dst_stride_y,
                                        const JavaParamRef<jobject>& j_dst_u,
                                        jint dst_stride_u,
                                        const JavaParamRef<jobject>& j_dst_v,
                                        jint dst_stride_v) {
  const int src_stride_y = src_width;
  const int src_stride_uv = src_width;
  const int crop_chroma_x = crop_x / 2;
  const int crop_chroma_y = crop_y / 2;

  jboolean was_copy;
  jbyte* src_bytes = jni->GetByteArrayElements(j_src.obj(), &was_copy);
  RTC_DCHECK(!was_copy);
  uint8_t const* src_y = reinterpret_cast<uint8_t const*>(src_bytes);
  uint8_t const* src_uv = src_y + src_height * src_stride_y;

  uint8_t* dst_y =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_y.obj()));
  uint8_t* dst_u =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_u.obj()));
  uint8_t* dst_v =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_v.obj()));

  // Crop using pointer arithmetic.
  src_y += crop_x + crop_y * src_stride_y;
  src_uv += 2 * crop_chroma_x + crop_chroma_y * src_stride_uv;

  NV12ToI420Scaler scaler;
  // U- and V-planes are swapped because this is NV21 not NV12.
  scaler.NV12ToI420Scale(src_y, src_stride_y, src_uv, src_stride_uv, crop_width,
                         crop_height, dst_y, dst_stride_y, dst_v, dst_stride_v,
                         dst_u, dst_stride_u, scale_width, scale_height);

  jni->ReleaseByteArrayElements(j_src.obj(), src_bytes, JNI_ABORT);
}

}  // namespace jni
}  // namespace webrtc
