/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
// This file contains the implementation of MediaStreamInterface interface.

#ifndef PC_MEDIA_STREAM_H_
#define PC_MEDIA_STREAM_H_

#include <string>

#include "api/media_stream_interface.h"
#include "api/notifier.h"
#include "api/scoped_refptr.h"

namespace webrtc {

class MediaStream : public Notifier<MediaStreamInterface> {
 public:
  static rtc::scoped_refptr<MediaStream> Create(const std::string& id);

  std::string id() const override { return id_; }

  bool AddTrack(rtc::scoped_refptr<AudioTrackInterface> track) override;
  bool AddTrack(rtc::scoped_refptr<VideoTrackInterface> track) override;
  bool RemoveTrack(rtc::scoped_refptr<AudioTrackInterface> track) override;
  bool RemoveTrack(rtc::scoped_refptr<VideoTrackInterface> track) override;
  rtc::scoped_refptr<AudioTrackInterface> FindAudioTrack(
      const std::string& track_id) override;
  rtc::scoped_refptr<VideoTrackInterface> FindVideoTrack(
      const std::string& track_id) override;

  AudioTrackVector GetAudioTracks() override { return audio_tracks_; }
  VideoTrackVector GetVideoTracks() override { return video_tracks_; }

 protected:
  explicit MediaStream(const std::string& id);

 private:
  template <typename TrackVector, typename Track>
  bool AddTrack(TrackVector* Tracks, rtc::scoped_refptr<Track> track);
  template <typename TrackVector>
  bool RemoveTrack(TrackVector* Tracks,
                   rtc::scoped_refptr<MediaStreamTrackInterface> track);

  const std::string id_;
  AudioTrackVector audio_tracks_;
  VideoTrackVector video_tracks_;
};

}  // namespace webrtc

#endif  // PC_MEDIA_STREAM_H_
