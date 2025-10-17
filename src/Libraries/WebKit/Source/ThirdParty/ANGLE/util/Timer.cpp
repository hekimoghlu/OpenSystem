/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
// Timer.cpp: Implementation of a high precision timer class.
//

#include "util/Timer.h"

#include "common/system_utils.h"

Timer::Timer() : mRunning(false), mStartTime(0), mStopTime(0) {}

void Timer::start()
{
    mStartTime    = angle::GetCurrentSystemTime();
    mStartCpuTime = angle::GetCurrentProcessCpuTime();
    mRunning      = true;
}

void Timer::stop()
{
    mStopTime    = angle::GetCurrentSystemTime();
    mStopCpuTime = angle::GetCurrentProcessCpuTime();
    mRunning     = false;
}

double Timer::getElapsedWallClockTime() const
{
    double endTime;
    if (mRunning)
    {
        endTime = angle::GetCurrentSystemTime();
    }
    else
    {
        endTime = mStopTime;
    }

    return endTime - mStartTime;
}

double Timer::getElapsedCpuTime() const
{
    double endTime;
    if (mRunning)
    {
        endTime = angle::GetCurrentProcessCpuTime();
    }
    else
    {
        endTime = mStopCpuTime;
    }

    return endTime - mStartCpuTime;
}
