/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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
#ifndef PC_MEDIA_STREAM_OBSERVER_H_
#define PC_MEDIA_STREAM_OBSERVER_H_

#include <functional>

#include "api/media_stream_interface.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// Helper class which will listen for changes to a stream and emit the
// corresponding signals.
class MediaStreamObserver : public ObserverInterface {
 public:
  explicit MediaStreamObserver(
      MediaStreamInterface* stream,
      std::function<void(AudioTrackInterface*, MediaStreamInterface*)>
          audio_track_added_callback,
      std::function<void(AudioTrackInterface*, MediaStreamInterface*)>
          audio_track_removed_callback,
      std::function<void(VideoTrackInterface*, MediaStreamInterface*)>
          video_track_added_callback,
      std::function<void(VideoTrackInterface*, MediaStreamInterface*)>
          video_track_removed_callback);
  ~MediaStreamObserver() override;

  const MediaStreamInterface* stream() const { return stream_.get(); }

  void OnChanged() override;

 private:
  rtc::scoped_refptr<MediaStreamInterface> stream_;
  AudioTrackVector cached_audio_tracks_;
  VideoTrackVector cached_video_tracks_;
  const std::function<void(AudioTrackInterface*, MediaStreamInterface*)>
      audio_track_added_callback_;
  const std::function<void(AudioTrackInterface*, MediaStreamInterface*)>
      audio_track_removed_callback_;
  const std::function<void(VideoTrackInterface*, MediaStreamInterface*)>
      video_track_added_callback_;
  const std::function<void(VideoTrackInterface*, MediaStreamInterface*)>
      video_track_removed_callback_;
};

}  // namespace webrtc

#endif  // PC_MEDIA_STREAM_OBSERVER_H_
