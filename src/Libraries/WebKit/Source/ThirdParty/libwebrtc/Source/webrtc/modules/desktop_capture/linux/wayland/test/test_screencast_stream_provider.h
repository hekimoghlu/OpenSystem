/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_TEST_TEST_SCREENCAST_STREAM_PROVIDER_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_TEST_TEST_SCREENCAST_STREAM_PROVIDER_H_

#include <pipewire/pipewire.h>
#include <spa/param/video/format-utils.h>

#include "modules/desktop_capture/linux/wayland/screencast_stream_utils.h"
#include "modules/desktop_capture/rgba_color.h"
#include "rtc_base/random.h"

namespace webrtc {

class TestScreenCastStreamProvider {
 public:
  class Observer {
   public:
    virtual void OnBufferAdded() = 0;
    virtual void OnFrameRecorded() = 0;
    virtual void OnStreamReady(uint32_t stream_node_id) = 0;
    virtual void OnStartStreaming() = 0;
    virtual void OnStopStreaming() = 0;

   protected:
    Observer() = default;
    virtual ~Observer() = default;
  };

  enum FrameDefect { None, EmptyData, CorruptedData, CorruptedMetadata };

  explicit TestScreenCastStreamProvider(Observer* observer,
                                        uint32_t width,
                                        uint32_t height);
  ~TestScreenCastStreamProvider();

  uint32_t PipeWireNodeId();

  void RecordFrame(RgbaColor rgba_color, FrameDefect frame_defect = None);
  void StartStreaming();
  void StopStreaming();

 private:
  Observer* observer_;

  // Resolution parameters.
  uint32_t width_ = 0;
  uint32_t height_ = 0;

  bool is_streaming_ = false;
  uint32_t pw_node_id_ = 0;

  // PipeWire types
  struct pw_context* pw_context_ = nullptr;
  struct pw_core* pw_core_ = nullptr;
  struct pw_stream* pw_stream_ = nullptr;
  struct pw_thread_loop* pw_main_loop_ = nullptr;

  spa_hook spa_core_listener_;
  spa_hook spa_stream_listener_;

  // event handlers
  pw_core_events pw_core_events_ = {};
  pw_stream_events pw_stream_events_ = {};

  struct spa_video_info_raw spa_video_format_;

  // PipeWire callbacks
  static void OnCoreError(void* data,
                          uint32_t id,
                          int seq,
                          int res,
                          const char* message);
  static void OnStreamAddBuffer(void* data, pw_buffer* buffer);
  static void OnStreamRemoveBuffer(void* data, pw_buffer* buffer);
  static void OnStreamParamChanged(void* data,
                                   uint32_t id,
                                   const struct spa_pod* format);
  static void OnStreamStateChanged(void* data,
                                   pw_stream_state old_state,
                                   pw_stream_state state,
                                   const char* error_message);
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_WAYLAND_TEST_TEST_SCREENCAST_STREAM_PROVIDER_H_
