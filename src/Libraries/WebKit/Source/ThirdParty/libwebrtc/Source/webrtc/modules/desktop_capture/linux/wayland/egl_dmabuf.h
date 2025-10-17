/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_EGL_DMABUF_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_EGL_DMABUF_H_

#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <gbm.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "modules/desktop_capture/desktop_geometry.h"

namespace webrtc {

class EglDmaBuf {
 public:
  struct EGLStruct {
    std::vector<std::string> extensions;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
  };

  struct PlaneData {
    int32_t fd;
    uint32_t stride;
    uint32_t offset;
  };

  EglDmaBuf();
  ~EglDmaBuf();

  // Returns whether the image was successfully imported from
  // given DmaBuf and its parameters
  bool ImageFromDmaBuf(const DesktopSize& size,
                       uint32_t format,
                       const std::vector<PlaneData>& plane_datas,
                       uint64_t modifiers,
                       const DesktopVector& offset,
                       const DesktopSize& buffer_size,
                       uint8_t* data);
  std::vector<uint64_t> QueryDmaBufModifiers(uint32_t format);

  bool IsEglInitialized() const { return egl_initialized_; }

 private:
  bool GetClientExtensions(EGLDisplay dpy, EGLint name);

  bool egl_initialized_ = false;
  bool has_image_dma_buf_import_ext_ = false;
  int32_t drm_fd_ = -1;               // for GBM buffer mmap
  gbm_device* gbm_device_ = nullptr;  // for passed GBM buffer retrieval

  GLuint fbo_ = 0;
  GLuint texture_ = 0;
  EGLStruct egl_;

  std::optional<std::string> GetRenderNode();
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_EGL_DMABUF_H_
