/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
// Unit tests for ComfortNoise class.

#include "modules/audio_coding/neteq/comfort_noise.h"

#include "modules/audio_coding/neteq/mock/mock_decoder_database.h"
#include "modules/audio_coding/neteq/sync_buffer.h"
#include "test/gtest.h"

namespace webrtc {

TEST(ComfortNoise, CreateAndDestroy) {
  int fs = 8000;
  MockDecoderDatabase db;
  SyncBuffer sync_buffer(1, 1000);
  ComfortNoise cn(fs, &db, &sync_buffer);
  EXPECT_CALL(db, Die());  // Called when `db` goes out of scope.
}

// TODO(hlundin): Write more tests.

}  // namespace webrtc
