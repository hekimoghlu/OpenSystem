/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#ifndef PC_STREAM_COLLECTION_H_
#define PC_STREAM_COLLECTION_H_

#include <string>
#include <utility>
#include <vector>

#include "api/peer_connection_interface.h"

namespace webrtc {

// Implementation of StreamCollection.
class StreamCollection : public StreamCollectionInterface {
 public:
  static rtc::scoped_refptr<StreamCollection> Create() {
    return rtc::make_ref_counted<StreamCollection>();
  }

  static rtc::scoped_refptr<StreamCollection> Create(
      StreamCollection* streams) {
    return rtc::make_ref_counted<StreamCollection>(streams);
  }

  virtual size_t count() { return media_streams_.size(); }

  virtual MediaStreamInterface* at(size_t index) {
    return media_streams_.at(index).get();
  }

  virtual MediaStreamInterface* find(const std::string& id) {
    for (StreamVector::iterator it = media_streams_.begin();
         it != media_streams_.end(); ++it) {
      if ((*it)->id().compare(id) == 0) {
        return (*it).get();
      }
    }
    return NULL;
  }

  virtual MediaStreamTrackInterface* FindAudioTrack(const std::string& id) {
    for (size_t i = 0; i < media_streams_.size(); ++i) {
      MediaStreamTrackInterface* track =
          media_streams_[i]->FindAudioTrack(id).get();
      if (track) {
        return track;
      }
    }
    return NULL;
  }

  virtual MediaStreamTrackInterface* FindVideoTrack(const std::string& id) {
    for (size_t i = 0; i < media_streams_.size(); ++i) {
      MediaStreamTrackInterface* track =
          media_streams_[i]->FindVideoTrack(id).get();
      if (track) {
        return track;
      }
    }
    return NULL;
  }

  void AddStream(rtc::scoped_refptr<MediaStreamInterface> stream) {
    for (StreamVector::iterator it = media_streams_.begin();
         it != media_streams_.end(); ++it) {
      if ((*it)->id().compare(stream->id()) == 0)
        return;
    }
    media_streams_.push_back(std::move(stream));
  }

  void RemoveStream(MediaStreamInterface* remove_stream) {
    for (StreamVector::iterator it = media_streams_.begin();
         it != media_streams_.end(); ++it) {
      if ((*it)->id().compare(remove_stream->id()) == 0) {
        media_streams_.erase(it);
        break;
      }
    }
  }

 protected:
  StreamCollection() {}
  explicit StreamCollection(StreamCollection* original)
      : media_streams_(original->media_streams_) {}
  typedef std::vector<rtc::scoped_refptr<MediaStreamInterface> > StreamVector;
  StreamVector media_streams_;
};

}  // namespace webrtc

#endif  // PC_STREAM_COLLECTION_H_
