/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#ifndef MODULES_VIDEO_CAPTURE_LINUX_PIPEWIRE_SESSION_H_
#define MODULES_VIDEO_CAPTURE_LINUX_PIPEWIRE_SESSION_H_

#include <pipewire/core.h>
#include <pipewire/pipewire.h>

#include <deque>
#include <string>
#include <vector>

#include "api/ref_counted_base.h"
#include "api/scoped_refptr.h"
#include "modules/portal/pipewire_utils.h"
#include "modules/video_capture/linux/camera_portal.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_options.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace videocapturemodule {

class PipeWireSession;
class VideoCaptureModulePipeWire;

// PipeWireNode objects are the local representation of PipeWire node objects.
// The portal API ensured that only camera nodes are visible to the client.
// So they all represent one camera that is available via PipeWire.
class PipeWireNode {
 public:
  struct PipeWireNodeDeleter {
    void operator()(PipeWireNode* node) const noexcept;
  };

  using PipeWireNodePtr =
      std::unique_ptr<PipeWireNode, PipeWireNode::PipeWireNodeDeleter>;
  static PipeWireNodePtr Create(PipeWireSession* session,
                                uint32_t id,
                                const spa_dict* props);

  uint32_t id() const { return id_; }
  std::string display_name() const { return display_name_; }
  std::string unique_id() const { return unique_id_; }
  std::string model_id() const { return model_id_; }
  std::vector<VideoCaptureCapability> capabilities() const {
    return capabilities_;
  }

 protected:
  PipeWireNode(PipeWireSession* session, uint32_t id, const spa_dict* props);

 private:
  static void OnNodeInfo(void* data, const pw_node_info* info);
  static void OnNodeParam(void* data,
                          int seq,
                          uint32_t id,
                          uint32_t index,
                          uint32_t next,
                          const spa_pod* param);
  static bool ParseFormat(const spa_pod* param, VideoCaptureCapability* cap);

  pw_proxy* proxy_;
  spa_hook node_listener_;
  PipeWireSession* session_;
  uint32_t id_;
  std::string display_name_;
  std::string unique_id_;
  std::string model_id_;
  std::vector<VideoCaptureCapability> capabilities_;
};

class CameraPortalNotifier : public CameraPortal::PortalNotifier {
 public:
  CameraPortalNotifier(PipeWireSession* session);
  ~CameraPortalNotifier() = default;

  void OnCameraRequestResult(xdg_portal::RequestResponse result,
                             int fd) override;

 private:
  PipeWireSession* session_;
};

class PipeWireSession : public rtc::RefCountedNonVirtual<PipeWireSession> {
 public:
  PipeWireSession();
  ~PipeWireSession();

  void Init(VideoCaptureOptions::Callback* callback,
            int fd = kInvalidPipeWireFd);
  const std::deque<PipeWireNode::PipeWireNodePtr>& nodes() const {
    return nodes_;
  }

  friend class CameraPortalNotifier;
  friend class PipeWireNode;
  friend class VideoCaptureModulePipeWire;

 private:
  void InitPipeWire(int fd);
  bool StartPipeWire(int fd);
  void StopPipeWire();
  void PipeWireSync();

  static void OnCoreError(void* data,
                          uint32_t id,
                          int seq,
                          int res,
                          const char* message);
  static void OnCoreDone(void* data, uint32_t id, int seq);

  static void OnRegistryGlobal(void* data,
                               uint32_t id,
                               uint32_t permissions,
                               const char* type,
                               uint32_t version,
                               const spa_dict* props);
  static void OnRegistryGlobalRemove(void* data, uint32_t id);

  void Finish(VideoCaptureOptions::Status status);
  void Cleanup();

  webrtc::Mutex callback_lock_;
  VideoCaptureOptions::Callback* callback_ RTC_GUARDED_BY(&callback_lock_) =
      nullptr;

  VideoCaptureOptions::Status status_;

  struct pw_thread_loop* pw_main_loop_ = nullptr;
  struct pw_context* pw_context_ = nullptr;
  struct pw_core* pw_core_ = nullptr;
  struct spa_hook core_listener_;

  struct pw_registry* pw_registry_ = nullptr;
  struct spa_hook registry_listener_;

  int sync_seq_ = 0;

  std::deque<PipeWireNode::PipeWireNodePtr> nodes_;
  std::unique_ptr<CameraPortal> portal_;
  std::unique_ptr<CameraPortalNotifier> portal_notifier_;
};

}  // namespace videocapturemodule
}  // namespace webrtc

#endif  // MODULES_VIDEO_CAPTURE_LINUX_PIPEWIRE_SESSION_H_
