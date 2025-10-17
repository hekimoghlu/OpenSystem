/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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

//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef SAMPLE_UTIL_TIMER_H
#define SAMPLE_UTIL_TIMER_H

class Timer final
{
  public:
    Timer();
    ~Timer() {}

    // Use start() and stop() to record the duration and use getElapsedWallClockTime() to query that
    // duration.  If getElapsedWallClockTime() is called in between, it will report the elapsed time
    // since start().
    void start();
    void stop();
    double getElapsedWallClockTime() const;
    double getElapsedCpuTime() const;

  private:
    bool mRunning;
    double mStartTime;
    double mStopTime;
    double mStartCpuTime;
    double mStopCpuTime;
};

#endif  // SAMPLE_UTIL_TIMER_H
