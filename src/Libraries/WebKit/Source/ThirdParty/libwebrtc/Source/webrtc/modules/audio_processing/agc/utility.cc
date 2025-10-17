/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#include "modules/audio_processing/agc/utility.h"

#include <math.h>

namespace webrtc {

static const double kLog10 = 2.30258509299;
static const double kLinear2DbScale = 20.0 / kLog10;
static const double kLinear2LoudnessScale = 13.4 / kLog10;

double Loudness2Db(double loudness) {
  return loudness * kLinear2DbScale / kLinear2LoudnessScale;
}

double Linear2Loudness(double rms) {
  if (rms == 0)
    return -15;
  return kLinear2LoudnessScale * log(rms);
}

double Db2Loudness(double db) {
  return db * kLinear2LoudnessScale / kLinear2DbScale;
}

double Dbfs2Loudness(double dbfs) {
  return Db2Loudness(90 + dbfs);
}

}  // namespace webrtc
