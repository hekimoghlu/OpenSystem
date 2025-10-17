/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#ifndef SRC_TRACK_ENTRY_PARSER_H_
#define SRC_TRACK_ENTRY_PARSER_H_

#include "src/audio_parser.h"
#include "src/bool_parser.h"
#include "src/byte_parser.h"
#include "src/content_encodings_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "src/video_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#TrackEntry
// http://www.webmproject.org/docs/container/#TrackEntry
class TrackEntryParser : public MasterValueParser<TrackEntry> {
 public:
  TrackEntryParser()
      : MasterValueParser<TrackEntry>(
            MakeChild<UnsignedIntParser>(Id::kTrackNumber,
                                         &TrackEntry::track_number),
            MakeChild<UnsignedIntParser>(Id::kTrackUid, &TrackEntry::track_uid),
            MakeChild<IntParser<TrackType>>(Id::kTrackType,
                                            &TrackEntry::track_type),
            MakeChild<BoolParser>(Id::kFlagEnabled, &TrackEntry::is_enabled),
            MakeChild<BoolParser>(Id::kFlagDefault, &TrackEntry::is_default),
            MakeChild<BoolParser>(Id::kFlagForced, &TrackEntry::is_forced),
            MakeChild<BoolParser>(Id::kFlagLacing, &TrackEntry::uses_lacing),
            MakeChild<UnsignedIntParser>(Id::kDefaultDuration,
                                         &TrackEntry::default_duration),
            MakeChild<StringParser>(Id::kName, &TrackEntry::name),
            MakeChild<StringParser>(Id::kLanguage, &TrackEntry::language),
            MakeChild<StringParser>(Id::kCodecId, &TrackEntry::codec_id),
            MakeChild<BinaryParser>(Id::kCodecPrivate,
                                    &TrackEntry::codec_private),
            MakeChild<StringParser>(Id::kCodecName, &TrackEntry::codec_name),
            MakeChild<UnsignedIntParser>(Id::kCodecDelay,
                                         &TrackEntry::codec_delay),
            MakeChild<UnsignedIntParser>(Id::kSeekPreRoll,
                                         &TrackEntry::seek_pre_roll),
            MakeChild<VideoParser>(Id::kVideo, &TrackEntry::video),
            MakeChild<AudioParser>(Id::kAudio, &TrackEntry::audio),
            MakeChild<ContentEncodingsParser>(
                Id::kContentEncodings, &TrackEntry::content_encodings)) {}

 protected:
  Status OnParseCompleted(Callback* callback) override {
    return callback->OnTrackEntry(metadata(Id::kTrackEntry), value());
  }
};

}  // namespace webm

#endif  // SRC_TRACK_ENTRY_PARSER_H_
