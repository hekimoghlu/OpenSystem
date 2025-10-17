/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#ifndef RTC_BASE_TEST_UTILS_H_
#define RTC_BASE_TEST_UTILS_H_

// Utilities for testing rtc infrastructure in unittests

#include <map>
#include <utility>

#include "rtc_base/socket.h"
#include "rtc_base/third_party/sigslot/sigslot.h"

namespace webrtc {
namespace testing {

///////////////////////////////////////////////////////////////////////////////
// StreamSink - Monitor asynchronously signalled events from Socket.
///////////////////////////////////////////////////////////////////////////////

// Note: Any event that is an error is treated as SSE_ERROR instead of that
// event.

enum StreamSinkEvent {
  SSE_OPEN = 1,
  SSE_READ = 2,
  SSE_WRITE = 4,
  SSE_CLOSE = 8,
  SSE_ERROR = 16
};

class StreamSink : public sigslot::has_slots<> {
 public:
  StreamSink();
  ~StreamSink() override;

  void Monitor(rtc::Socket* socket) {
    socket->SignalConnectEvent.connect(this, &StreamSink::OnConnectEvent);
    socket->SignalReadEvent.connect(this, &StreamSink::OnReadEvent);
    socket->SignalWriteEvent.connect(this, &StreamSink::OnWriteEvent);
    socket->SignalCloseEvent.connect(this, &StreamSink::OnCloseEvent);
    // In case you forgot to unmonitor a previous object with this address
    events_.erase(socket);
  }
  void Unmonitor(rtc::Socket* socket) {
    socket->SignalConnectEvent.disconnect(this);
    socket->SignalReadEvent.disconnect(this);
    socket->SignalWriteEvent.disconnect(this);
    socket->SignalCloseEvent.disconnect(this);
    events_.erase(socket);
  }
  bool Check(rtc::Socket* socket, StreamSinkEvent event, bool reset = true) {
    return DoCheck(socket, event, reset);
  }

 private:
  typedef std::map<rtc::Socket*, int> EventMap;

  void OnConnectEvent(rtc::Socket* socket) { AddEvents(socket, SSE_OPEN); }
  void OnReadEvent(rtc::Socket* socket) { AddEvents(socket, SSE_READ); }
  void OnWriteEvent(rtc::Socket* socket) { AddEvents(socket, SSE_WRITE); }
  void OnCloseEvent(rtc::Socket* socket, int error) {
    AddEvents(socket, (0 == error) ? SSE_CLOSE : SSE_ERROR);
  }

  void AddEvents(rtc::Socket* obj, int events) {
    EventMap::iterator it = events_.find(obj);
    if (events_.end() == it) {
      events_.insert(EventMap::value_type(obj, events));
    } else {
      it->second |= events;
    }
  }
  bool DoCheck(rtc::Socket* obj, StreamSinkEvent event, bool reset) {
    EventMap::iterator it = events_.find(obj);
    if ((events_.end() == it) || (0 == (it->second & event))) {
      return false;
    }
    if (reset) {
      it->second &= ~event;
    }
    return true;
  }

  EventMap events_;
};

}  // namespace testing
}  // namespace webrtc

#endif  // RTC_BASE_TEST_UTILS_H_
