/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#ifndef MODULES_AUDIO_CODING_TEST_RTPFILE_H_
#define MODULES_AUDIO_CODING_TEST_RTPFILE_H_

#include <stdio.h>

#include <queue>

#include "absl/strings/string_view.h"
#include "api/rtp_headers.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class RTPStream {
 public:
  virtual ~RTPStream() {}

  virtual void Write(uint8_t payloadType,
                     uint32_t timeStamp,
                     int16_t seqNo,
                     const uint8_t* payloadData,
                     size_t payloadSize,
                     uint32_t frequency) = 0;

  // Returns the packet's payload size. Zero should be treated as an
  // end-of-stream (in the case that EndOfFile() is true) or an error.
  virtual size_t Read(RTPHeader* rtp_Header,
                      uint8_t* payloadData,
                      size_t payloadSize,
                      uint32_t* offset) = 0;
  virtual bool EndOfFile() const = 0;

 protected:
  void MakeRTPheader(uint8_t* rtpHeader,
                     uint8_t payloadType,
                     int16_t seqNo,
                     uint32_t timeStamp,
                     uint32_t ssrc);

  void ParseRTPHeader(RTPHeader* rtp_header, const uint8_t* rtpHeader);
};

class RTPPacket {
 public:
  RTPPacket(uint8_t payloadType,
            uint32_t timeStamp,
            int16_t seqNo,
            const uint8_t* payloadData,
            size_t payloadSize,
            uint32_t frequency);

  ~RTPPacket();

  uint8_t payloadType;
  uint32_t timeStamp;
  int16_t seqNo;
  uint8_t* payloadData;
  size_t payloadSize;
  uint32_t frequency;
};

class RTPBuffer : public RTPStream {
 public:
  RTPBuffer() = default;

  ~RTPBuffer() = default;

  void Write(uint8_t payloadType,
             uint32_t timeStamp,
             int16_t seqNo,
             const uint8_t* payloadData,
             size_t payloadSize,
             uint32_t frequency) override;

  size_t Read(RTPHeader* rtp_header,
              uint8_t* payloadData,
              size_t payloadSize,
              uint32_t* offset) override;

  bool EndOfFile() const override;

 private:
  mutable Mutex mutex_;
  std::queue<RTPPacket*> _rtpQueue RTC_GUARDED_BY(&mutex_);
};

class RTPFile : public RTPStream {
 public:
  ~RTPFile() {}

  RTPFile() : _rtpFile(NULL), _rtpEOF(false) {}

  void Open(absl::string_view outFilename, absl::string_view mode);

  void Close();

  void WriteHeader();

  void ReadHeader();

  void Write(uint8_t payloadType,
             uint32_t timeStamp,
             int16_t seqNo,
             const uint8_t* payloadData,
             size_t payloadSize,
             uint32_t frequency) override;

  size_t Read(RTPHeader* rtp_header,
              uint8_t* payloadData,
              size_t payloadSize,
              uint32_t* offset) override;

  bool EndOfFile() const override { return _rtpEOF; }

 private:
  FILE* _rtpFile;
  bool _rtpEOF;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_TEST_RTPFILE_H_
