/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
// This file contains an interface for the (obsolete) StatsCollector class that
// is used by compilation units that do not wish to depend on the StatsCollector
// implementation.

#ifndef PC_LEGACY_STATS_COLLECTOR_INTERFACE_H_
#define PC_LEGACY_STATS_COLLECTOR_INTERFACE_H_

#include <stdint.h>

#include "api/legacy_stats_types.h"
#include "api/media_stream_interface.h"

namespace webrtc {

class LegacyStatsCollectorInterface {
 public:
  virtual ~LegacyStatsCollectorInterface() {}

  // Adds a local audio track that is used for getting some voice statistics.
  virtual void AddLocalAudioTrack(AudioTrackInterface* audio_track,
                                  uint32_t ssrc) = 0;

  // Removes a local audio tracks that is used for getting some voice
  // statistics.
  virtual void RemoveLocalAudioTrack(AudioTrackInterface* audio_track,
                                     uint32_t ssrc) = 0;
  virtual void GetStats(MediaStreamTrackInterface* track,
                        StatsReports* reports) = 0;
};

}  // namespace webrtc

#endif  // PC_LEGACY_STATS_COLLECTOR_INTERFACE_H_
