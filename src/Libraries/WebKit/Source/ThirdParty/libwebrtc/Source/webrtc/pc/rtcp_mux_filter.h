/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#ifndef PC_RTCP_MUX_FILTER_H_
#define PC_RTCP_MUX_FILTER_H_

#include "pc/session_description.h"

namespace cricket {

// RTCP Muxer, as defined in RFC 5761 (http://tools.ietf.org/html/rfc5761)
class RtcpMuxFilter {
 public:
  RtcpMuxFilter();

  // Whether RTCP mux has been negotiated with a final answer (not provisional).
  bool IsFullyActive() const;

  // Whether RTCP mux has been negotiated with a provisional answer; this means
  // a later answer could disable RTCP mux, and so the RTCP transport should
  // not be disposed yet.
  bool IsProvisionallyActive() const;

  // Whether the filter is active, i.e. has RTCP mux been properly negotiated,
  // either with a final or provisional answer.
  bool IsActive() const;

  // Make the filter active (fully, not provisionally) regardless of the
  // current state. This should be used when an endpoint *requires* RTCP mux.
  void SetActive();

  // Specifies whether the offer indicates the use of RTCP mux.
  bool SetOffer(bool offer_enable, ContentSource src);

  // Specifies whether the provisional answer indicates the use of RTCP mux.
  bool SetProvisionalAnswer(bool answer_enable, ContentSource src);

  // Specifies whether the answer indicates the use of RTCP mux.
  bool SetAnswer(bool answer_enable, ContentSource src);

 private:
  bool ExpectOffer(bool offer_enable, ContentSource source);
  bool ExpectAnswer(ContentSource source);
  enum State {
    // RTCP mux filter unused.
    ST_INIT,
    // Offer with RTCP mux enabled received.
    // RTCP mux filter is not active.
    ST_RECEIVEDOFFER,
    // Offer with RTCP mux enabled sent.
    // RTCP mux filter can demux incoming packets but is not active.
    ST_SENTOFFER,
    // RTCP mux filter is active but the sent answer is only provisional.
    // When the final answer is set, the state transitions to ST_ACTIVE or
    // ST_INIT.
    ST_SENTPRANSWER,
    // RTCP mux filter is active but the received answer is only provisional.
    // When the final answer is set, the state transitions to ST_ACTIVE or
    // ST_INIT.
    ST_RECEIVEDPRANSWER,
    // Offer and answer set, RTCP mux enabled. It is not possible to de-activate
    // the filter.
    ST_ACTIVE
  };
  State state_;
  bool offer_enable_;
};

}  // namespace cricket

#endif  // PC_RTCP_MUX_FILTER_H_
