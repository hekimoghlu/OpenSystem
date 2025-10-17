/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
// Unit tests for DelayManager class.

#include "modules/audio_coding/neteq/delay_manager.h"

#include "api/neteq/tick_timer.h"
#include "test/explicit_key_value_config.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using test::ExplicitKeyValueConfig;

TEST(DelayManagerTest, UpdateNormal) {
  TickTimer tick_timer;
  DelayManager dm(DelayManager::Config(ExplicitKeyValueConfig("")),
                  &tick_timer);
  for (int i = 0; i < 50; ++i) {
    dm.Update(0, false);
    tick_timer.Increment(2);
  }
  EXPECT_EQ(20, dm.TargetDelayMs());
}

}  // namespace
}  // namespace webrtc
