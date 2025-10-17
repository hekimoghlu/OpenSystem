/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_DTMF_BUFFER_H_
#define MODULES_AUDIO_CODING_NETEQ_DTMF_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include <list>

namespace webrtc {

struct DtmfEvent {
  uint32_t timestamp;
  int event_no;
  int volume;
  int duration;
  bool end_bit;

  // Constructors
  DtmfEvent()
      : timestamp(0), event_no(0), volume(0), duration(0), end_bit(false) {}
  DtmfEvent(uint32_t ts, int ev, int vol, int dur, bool end)
      : timestamp(ts), event_no(ev), volume(vol), duration(dur), end_bit(end) {}
};

// This is the buffer holding DTMF events while waiting for them to be played.
class DtmfBuffer {
 public:
  enum BufferReturnCodes {
    kOK = 0,
    kInvalidPointer,
    kPayloadTooShort,
    kInvalidEventParameters,
    kInvalidSampleRate
  };

  // Set up the buffer for use at sample rate `fs_hz`.
  explicit DtmfBuffer(int fs_hz);

  virtual ~DtmfBuffer();

  DtmfBuffer(const DtmfBuffer&) = delete;
  DtmfBuffer& operator=(const DtmfBuffer&) = delete;

  // Flushes the buffer.
  virtual void Flush();

  // Static method to parse 4 bytes from `payload` as a DTMF event (RFC 4733)
  // and write the parsed information into the struct `event`. Input variable
  // `rtp_timestamp` is simply copied into the struct.
  static int ParseEvent(uint32_t rtp_timestamp,
                        const uint8_t* payload,
                        size_t payload_length_bytes,
                        DtmfEvent* event);

  // Inserts `event` into the buffer. The method looks for a matching event and
  // merges the two if a match is found.
  virtual int InsertEvent(const DtmfEvent& event);

  // Checks if a DTMF event should be played at time `current_timestamp`. If so,
  // the method returns true; otherwise false. The parameters of the event to
  // play will be written to `event`.
  virtual bool GetEvent(uint32_t current_timestamp, DtmfEvent* event);

  // Number of events in the buffer.
  virtual size_t Length() const;

  virtual bool Empty() const;

  // Set a new sample rate.
  virtual int SetSampleRate(int fs_hz);

 private:
  typedef std::list<DtmfEvent> DtmfList;

  int max_extrapolation_samples_;
  int frame_len_samples_;  // TODO(hlundin): Remove this later.

  // Compares two events and returns true if they are the same.
  static bool SameEvent(const DtmfEvent& a, const DtmfEvent& b);

  // Merges `event` to the event pointed out by `it`. The method checks that
  // the two events are the same (using the SameEvent method), and merges them
  // if that was the case, returning true. If the events are not the same, false
  // is returned.
  bool MergeEvents(DtmfList::iterator it, const DtmfEvent& event);

  // Method used by the sort algorithm to rank events in the buffer.
  static bool CompareEvents(const DtmfEvent& a, const DtmfEvent& b);

  DtmfList buffer_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_DTMF_BUFFER_H_
