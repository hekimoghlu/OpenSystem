/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#ifndef API_TEST_MOCK_VIDEO_TRACK_H_
#define API_TEST_MOCK_VIDEO_TRACK_H_

#include <string>

#include "api/media_stream_interface.h"
#include "api/scoped_refptr.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "rtc_base/ref_counted_object.h"
#include "test/gmock.h"

namespace webrtc {

class MockVideoTrack
    : public rtc::RefCountedObject<webrtc::VideoTrackInterface> {
 public:
  static rtc::scoped_refptr<MockVideoTrack> Create() {
    return rtc::scoped_refptr<MockVideoTrack>(new MockVideoTrack());
  }

  // NotifierInterface
  MOCK_METHOD(void,
              RegisterObserver,
              (ObserverInterface * observer),
              (override));
  MOCK_METHOD(void,
              UnregisterObserver,
              (ObserverInterface * observer),
              (override));

  // MediaStreamTrackInterface
  MOCK_METHOD(std::string, kind, (), (const, override));
  MOCK_METHOD(std::string, id, (), (const, override));
  MOCK_METHOD(bool, enabled, (), (const, override));
  MOCK_METHOD(bool, set_enabled, (bool enable), (override));
  MOCK_METHOD(TrackState, state, (), (const, override));

  // VideoSourceInterface
  MOCK_METHOD(void,
              AddOrUpdateSink,
              (rtc::VideoSinkInterface<VideoFrame> * sink,
               const rtc::VideoSinkWants& wants),
              (override));
  // RemoveSink must guarantee that at the time the method returns,
  // there is no current and no future calls to VideoSinkInterface::OnFrame.
  MOCK_METHOD(void,
              RemoveSink,
              (rtc::VideoSinkInterface<VideoFrame> * sink),
              (override));

  // VideoTrackInterface
  MOCK_METHOD(VideoTrackSourceInterface*, GetSource, (), (const, override));

  MOCK_METHOD(ContentHint, content_hint, (), (const, override));
  MOCK_METHOD(void, set_content_hint, (ContentHint hint), (override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_VIDEO_TRACK_H_
