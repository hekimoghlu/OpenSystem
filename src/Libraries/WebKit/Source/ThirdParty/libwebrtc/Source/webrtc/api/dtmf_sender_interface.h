/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#ifndef API_DTMF_SENDER_INTERFACE_H_
#define API_DTMF_SENDER_INTERFACE_H_

#include <string>

#include "api/ref_count.h"

namespace webrtc {

// DtmfSender callback interface, used to implement RTCDtmfSender events.
// Applications should implement this interface to get notifications from the
// DtmfSender.
class DtmfSenderObserverInterface {
 public:
  // Triggered when DTMF `tone` is sent.
  // If `tone` is empty that means the DtmfSender has sent out all the given
  // tones.
  // The callback includes the state of the tone buffer at the time when
  // the tone finished playing.
  virtual void OnToneChange(const std::string& /* tone */,
                            const std::string& /* tone_buffer */) {}
  // DEPRECATED: Older API without tone buffer.
  // TODO(bugs.webrtc.org/9725): Remove old API and default implementation
  // when old callers are gone.
  virtual void OnToneChange(const std::string& /* tone */) {}

 protected:
  virtual ~DtmfSenderObserverInterface() = default;
};

// The interface of native implementation of the RTCDTMFSender defined by the
// WebRTC W3C Editor's Draft.
// See: https://www.w3.org/TR/webrtc/#peer-to-peer-dtmf
class DtmfSenderInterface : public webrtc::RefCountInterface {
 public:
  // Provides the spec compliant default 2 second delay for the ',' character.
  static const int kDtmfDefaultCommaDelayMs = 2000;

  // Used to receive events from the DTMF sender. Only one observer can be
  // registered at a time. UnregisterObserver should be called before the
  // observer object is destroyed.
  virtual void RegisterObserver(DtmfSenderObserverInterface* observer) = 0;
  virtual void UnregisterObserver() = 0;

  // Returns true if this DtmfSender is capable of sending DTMF. Otherwise
  // returns false. To be able to send DTMF, the associated RtpSender must be
  // able to send packets, and a "telephone-event" codec must be negotiated.
  virtual bool CanInsertDtmf() = 0;

  // Queues a task that sends the DTMF `tones`. The `tones` parameter is treated
  // as a series of characters. The characters 0 through 9, A through D, #, and
  // * generate the associated DTMF tones. The characters a to d are equivalent
  // to A to D. The character ',' indicates a delay of 2 seconds before
  // processing the next character in the tones parameter.
  //
  // Unrecognized characters are ignored.
  //
  // The `duration` parameter indicates the duration in ms to use for each
  // character passed in the `tones` parameter. The duration cannot be more
  // than 6000 or less than 70.
  //
  // The `inter_tone_gap` parameter indicates the gap between tones in ms. The
  // `inter_tone_gap` must be at least 50 ms but should be as short as
  // possible.
  //
  // The `comma_delay` parameter indicates the delay after the ','
  // character. InsertDtmf specifies `comma_delay` as an argument
  // with a default value of 2 seconds as per the WebRTC spec. This parameter
  // allows users to comply with legacy WebRTC clients. The `comma_delay`
  // must be at least 50 ms.
  //
  // If InsertDtmf is called on the same object while an existing task for this
  // object to generate DTMF is still running, the previous task is canceled.
  // Returns true on success and false on failure.
  virtual bool InsertDtmf(const std::string& tones,
                          int duration,
                          int inter_tone_gap) {
    return InsertDtmf(tones, duration, inter_tone_gap,
                      kDtmfDefaultCommaDelayMs);
  }
  virtual bool InsertDtmf(const std::string& tones,
                          int duration,
                          int inter_tone_gap,
                          int /* comma_delay */) {
    // TODO(bugs.webrtc.org/165700): Remove once downstream implementations
    // override this signature rather than the 3-parameter one.
    return InsertDtmf(tones, duration, inter_tone_gap);
  }

  // Returns the tones remaining to be played out.
  virtual std::string tones() const = 0;

  // Returns the current tone duration value in ms.
  // This value will be the value last set via the InsertDtmf() method, or the
  // default value of 100 ms if InsertDtmf() was never called.
  virtual int duration() const = 0;

  // Returns the current value of the between-tone gap in ms.
  // This value will be the value last set via the InsertDtmf() method, or the
  // default value of 50 ms if InsertDtmf() was never called.
  virtual int inter_tone_gap() const = 0;

  // Returns the current value of the "," character delay in ms.
  // This value will be the value last set via the InsertDtmf() method, or the
  // default value of 2000 ms if InsertDtmf() was never called.
  virtual int comma_delay() const { return kDtmfDefaultCommaDelayMs; }

 protected:
  ~DtmfSenderInterface() override = default;
};

}  // namespace webrtc

#endif  // API_DTMF_SENDER_INTERFACE_H_
