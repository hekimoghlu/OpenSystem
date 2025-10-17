/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#ifndef API_TEST_MOCK_MEDIA_STREAM_INTERFACE_H_
#define API_TEST_MOCK_MEDIA_STREAM_INTERFACE_H_

#include <string>
#include <type_traits>

#include "api/audio_options.h"
#include "api/media_stream_interface.h"
#include "api/scoped_refptr.h"
#include "rtc_base/ref_counted_object.h"
#include "test/gmock.h"

namespace webrtc {

class MockAudioSource : public rtc::RefCountedObject<AudioSourceInterface> {
 public:
  static rtc::scoped_refptr<MockAudioSource> Create() {
    return rtc::scoped_refptr<MockAudioSource>(new MockAudioSource());
  }

  MOCK_METHOD(void,
              RegisterObserver,
              (ObserverInterface * observer),
              (override));
  MOCK_METHOD(void,
              UnregisterObserver,
              (ObserverInterface * observer),
              (override));
  MOCK_METHOD(SourceState, state, (), (const, override));
  MOCK_METHOD(bool, remote, (), (const, override));
  MOCK_METHOD(void, SetVolume, (double volume), (override));
  MOCK_METHOD(void,
              RegisterAudioObserver,
              (AudioObserver * observer),
              (override));
  MOCK_METHOD(void,
              UnregisterAudioObserver,
              (AudioObserver * observer),
              (override));
  MOCK_METHOD(void, AddSink, (AudioTrackSinkInterface * sink), (override));
  MOCK_METHOD(void, RemoveSink, (AudioTrackSinkInterface * sink), (override));
  MOCK_METHOD(const cricket::AudioOptions, options, (), (const, override));

 private:
  MockAudioSource() = default;
};

class MockAudioTrack : public rtc::RefCountedObject<AudioTrackInterface> {
 public:
  static rtc::scoped_refptr<MockAudioTrack> Create() {
    return rtc::scoped_refptr<MockAudioTrack>(new MockAudioTrack());
  }

  MOCK_METHOD(void,
              RegisterObserver,
              (ObserverInterface * observer),
              (override));
  MOCK_METHOD(void,
              UnregisterObserver,
              (ObserverInterface * observer),
              (override));
  MOCK_METHOD(std::string, kind, (), (const, override));
  MOCK_METHOD(std::string, id, (), (const, override));
  MOCK_METHOD(bool, enabled, (), (const, override));
  MOCK_METHOD(bool, set_enabled, (bool enable), (override));
  MOCK_METHOD(TrackState, state, (), (const, override));
  MOCK_METHOD(AudioSourceInterface*, GetSource, (), (const, override));
  MOCK_METHOD(void, AddSink, (AudioTrackSinkInterface * sink), (override));
  MOCK_METHOD(void, RemoveSink, (AudioTrackSinkInterface * sink), (override));
  MOCK_METHOD(bool, GetSignalLevel, (int* level), (override));
  MOCK_METHOD(rtc::scoped_refptr<AudioProcessorInterface>,
              GetAudioProcessor,
              (),
              (override));

 private:
  MockAudioTrack() = default;
};

class MockMediaStream : public MediaStreamInterface {
 public:
  MOCK_METHOD(std::string, id, (), (const, override));
  MOCK_METHOD(AudioTrackVector, GetAudioTracks, (), (override));
  MOCK_METHOD(VideoTrackVector, GetVideoTracks, (), (override));
  MOCK_METHOD(rtc::scoped_refptr<AudioTrackInterface>,
              FindAudioTrack,
              (const std::string& track_id),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<VideoTrackInterface>,
              FindVideoTrack,
              (const std::string& track_id),
              (override));
  MOCK_METHOD(bool,
              AddTrack,
              (rtc::scoped_refptr<AudioTrackInterface> track),
              (override));
  MOCK_METHOD(bool,
              AddTrack,
              (rtc::scoped_refptr<VideoTrackInterface> track),
              (override));
  MOCK_METHOD(bool,
              RemoveTrack,
              (rtc::scoped_refptr<AudioTrackInterface> track),
              (override));
  MOCK_METHOD(bool,
              RemoveTrack,
              (rtc::scoped_refptr<VideoTrackInterface> track),
              (override));
  MOCK_METHOD(void,
              RegisterObserver,
              (ObserverInterface * observer),
              (override));
  MOCK_METHOD(void,
              UnregisterObserver,
              (ObserverInterface * observer),
              (override));
};

static_assert(!std::is_abstract_v<rtc::RefCountedObject<MockMediaStream>>, "");

}  // namespace webrtc

#endif  // API_TEST_MOCK_MEDIA_STREAM_INTERFACE_H_
