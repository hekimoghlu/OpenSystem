/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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

#include "rtc_base/checks.h"
#include "sdk/android/generated_video_jni/NV12Buffer_jni.h"
#include "third_party/libyuv/include/libyuv/convert.h"
#include "third_party/libyuv/include/libyuv/scale.h"

namespace webrtc {
namespace jni {

static void JNI_NV12Buffer_CropAndScale(JNIEnv* jni,
                                        jint crop_x,
                                        jint crop_y,
                                        jint crop_width,
                                        jint crop_height,
                                        jint scale_width,
                                        jint scale_height,
                                        const JavaParamRef<jobject>& j_src,
                                        jint src_width,
                                        jint src_height,
                                        jint src_stride,
                                        jint src_slice_height,
                                        const JavaParamRef<jobject>& j_dst_y,
                                        jint dst_stride_y,
                                        const JavaParamRef<jobject>& j_dst_u,
                                        jint dst_stride_u,
                                        const JavaParamRef<jobject>& j_dst_v,
                                        jint dst_stride_v) {
  const int src_stride_y = src_stride;
  const int src_stride_uv = src_stride;
  const int crop_chroma_x = crop_x / 2;
  const int crop_chroma_y = crop_y / 2;
  const int crop_chroma_width = (crop_width + 1) / 2;
  const int crop_chroma_height = (crop_height + 1) / 2;
  const int tmp_stride_u = crop_chroma_width;
  const int tmp_stride_v = crop_chroma_width;
  const int tmp_size = crop_chroma_height * (tmp_stride_u + tmp_stride_v);

  uint8_t const* src_y =
      static_cast<uint8_t const*>(jni->GetDirectBufferAddress(j_src.obj()));
  uint8_t const* src_uv = src_y + src_slice_height * src_stride_y;

  uint8_t* dst_y =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_y.obj()));
  uint8_t* dst_u =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_u.obj()));
  uint8_t* dst_v =
      static_cast<uint8_t*>(jni->GetDirectBufferAddress(j_dst_v.obj()));

  // Crop using pointer arithmetic.
  src_y += crop_x + crop_y * src_stride_y;
  src_uv += 2 * crop_chroma_x + crop_chroma_y * src_stride_uv;

  std::vector<uint8_t> tmp_buffer(tmp_size);
  uint8_t* tmp_u = tmp_buffer.data();
  uint8_t* tmp_v = tmp_u + crop_chroma_height * tmp_stride_u;

  libyuv::SplitUVPlane(src_uv, src_stride_uv, tmp_u, tmp_stride_u, tmp_v,
                       tmp_stride_v, crop_chroma_width, crop_chroma_height);

  libyuv::I420Scale(src_y, src_stride_y, tmp_u, tmp_stride_u, tmp_v,
                    tmp_stride_v, crop_width, crop_height, dst_y, dst_stride_y,
                    dst_u, dst_stride_u, dst_v, dst_stride_v, scale_width,
                    scale_height, libyuv::kFilterBox);
}

}  // namespace jni
}  // namespace webrtc
