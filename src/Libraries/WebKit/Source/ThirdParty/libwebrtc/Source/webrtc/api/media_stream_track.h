/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#ifndef API_MEDIA_STREAM_TRACK_H_
#define API_MEDIA_STREAM_TRACK_H_

#include <string>

#include "absl/strings/string_view.h"
#include "api/media_stream_interface.h"
#include "api/notifier.h"

namespace webrtc {

// MediaTrack implements the interface common to AudioTrackInterface and
// VideoTrackInterface.
template <typename T>
class MediaStreamTrack : public Notifier<T> {
 public:
  typedef typename T::TrackState TypedTrackState;

  std::string id() const override { return id_; }
  MediaStreamTrackInterface::TrackState state() const override {
    return state_;
  }
  bool enabled() const override { return enabled_; }
  bool set_enabled(bool enable) override {
    bool fire_on_change = (enable != enabled_);
    enabled_ = enable;
    if (fire_on_change) {
      Notifier<T>::FireOnChanged();
    }
    return fire_on_change;
  }
  void set_ended() { set_state(MediaStreamTrackInterface::TrackState::kEnded); }

 protected:
  explicit MediaStreamTrack(absl::string_view id)
      : enabled_(true), id_(id), state_(MediaStreamTrackInterface::kLive) {}

  bool set_state(MediaStreamTrackInterface::TrackState new_state) {
    bool fire_on_change = (state_ != new_state);
    state_ = new_state;
    if (fire_on_change)
      Notifier<T>::FireOnChanged();
    return true;
  }

 private:
  bool enabled_;
  const std::string id_;
  MediaStreamTrackInterface::TrackState state_;
};

}  // namespace webrtc

#endif  // API_MEDIA_STREAM_TRACK_H_
