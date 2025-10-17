/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#include "pc/media_stream.h"

#include <stddef.h>

#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

template <class V>
static typename V::iterator FindTrack(V* vector, const std::string& track_id) {
  typename V::iterator it = vector->begin();
  for (; it != vector->end(); ++it) {
    if ((*it)->id() == track_id) {
      break;
    }
  }
  return it;
}

rtc::scoped_refptr<MediaStream> MediaStream::Create(const std::string& id) {
  return rtc::make_ref_counted<MediaStream>(id);
}

MediaStream::MediaStream(const std::string& id) : id_(id) {}

bool MediaStream::AddTrack(rtc::scoped_refptr<AudioTrackInterface> track) {
  return AddTrack<AudioTrackVector, AudioTrackInterface>(&audio_tracks_, track);
}

bool MediaStream::AddTrack(rtc::scoped_refptr<VideoTrackInterface> track) {
  return AddTrack<VideoTrackVector, VideoTrackInterface>(&video_tracks_, track);
}

bool MediaStream::RemoveTrack(rtc::scoped_refptr<AudioTrackInterface> track) {
  return RemoveTrack<AudioTrackVector>(&audio_tracks_, track);
}

bool MediaStream::RemoveTrack(rtc::scoped_refptr<VideoTrackInterface> track) {
  return RemoveTrack<VideoTrackVector>(&video_tracks_, track);
}

rtc::scoped_refptr<AudioTrackInterface> MediaStream::FindAudioTrack(
    const std::string& track_id) {
  AudioTrackVector::iterator it = FindTrack(&audio_tracks_, track_id);
  if (it == audio_tracks_.end())
    return nullptr;
  return *it;
}

rtc::scoped_refptr<VideoTrackInterface> MediaStream::FindVideoTrack(
    const std::string& track_id) {
  VideoTrackVector::iterator it = FindTrack(&video_tracks_, track_id);
  if (it == video_tracks_.end())
    return nullptr;
  return *it;
}

template <typename TrackVector, typename Track>
bool MediaStream::AddTrack(TrackVector* tracks,
                           rtc::scoped_refptr<Track> track) {
  typename TrackVector::iterator it = FindTrack(tracks, track->id());
  if (it != tracks->end())
    return false;
  tracks->emplace_back(std::move((track)));
  FireOnChanged();
  return true;
}

template <typename TrackVector>
bool MediaStream::RemoveTrack(
    TrackVector* tracks,
    rtc::scoped_refptr<MediaStreamTrackInterface> track) {
  RTC_DCHECK(tracks != NULL);
  if (!track)
    return false;
  typename TrackVector::iterator it = FindTrack(tracks, track->id());
  if (it == tracks->end())
    return false;
  tracks->erase(it);
  FireOnChanged();
  return true;
}

}  // namespace webrtc
