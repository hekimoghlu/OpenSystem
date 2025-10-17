/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#ifndef TEST_WIN_D3D_RENDERER_H_
#define TEST_WIN_D3D_RENDERER_H_

#include <Windows.h>
#include <d3d9.h>
#pragma comment(lib, "d3d9.lib")  // located in DirectX SDK

#include "api/scoped_refptr.h"
#include "test/video_renderer.h"

namespace webrtc {
namespace test {

class D3dRenderer : public VideoRenderer {
 public:
  static D3dRenderer* Create(const char* window_title,
                             size_t width,
                             size_t height);
  virtual ~D3dRenderer();

  void OnFrame(const webrtc::VideoFrame& frame) override;

 private:
  D3dRenderer(size_t width, size_t height);

  static LRESULT WINAPI WindowProc(HWND hwnd,
                                   UINT msg,
                                   WPARAM wparam,
                                   LPARAM lparam);
  bool Init(const char* window_title);
  void Resize(size_t width, size_t height);
  void Destroy();

  size_t width_, height_;

  HWND hwnd_;
  rtc::scoped_refptr<IDirect3D9> d3d_;
  rtc::scoped_refptr<IDirect3DDevice9> d3d_device_;

  rtc::scoped_refptr<IDirect3DTexture9> texture_;
  rtc::scoped_refptr<IDirect3DVertexBuffer9> vertex_buffer_;
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_WIN_D3D_RENDERER_H_
