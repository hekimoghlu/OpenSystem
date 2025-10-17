/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
// ANGLEPerfTestArgs.h:
//   Command line arguments for angle_perftests.
//

#ifndef TESTS_PERF_TESTS_ANGLE_PERF_TEST_ARGS_H_
#define TESTS_PERF_TESTS_ANGLE_PERF_TEST_ARGS_H_

#include <string>
#include <vector>
#include "common/Optional.h"

namespace angle
{
extern int gStepsPerTrial;
extern int gMaxStepsPerformed;
extern bool gEnableTrace;
extern const char *gTraceFile;
extern const char *gScreenshotDir;
extern bool gSaveScreenshots;
extern int gScreenshotFrame;
extern bool gRunToKeyFrame;
extern bool gVerboseLogging;
extern bool gWarmup;
extern int gTrialTimeSeconds;
extern int gTestTrials;
extern bool gNoFinish;
extern bool gRetraceMode;
extern bool gMinimizeGPUWork;
extern bool gTraceTestValidation;
extern const char *gTraceInterpreter;
extern const char *gPerfCounters;
extern const char *gUseANGLE;
extern const char *gUseGL;
extern bool gOffscreen;
extern bool gVsync;
extern int gFpsLimit;
extern const char *gPrintExtensionsToFile;
extern const char *gRequestedExtensions;
extern bool gIncludeInactiveResources;

// Constant for when trace's frame count should be used
constexpr int kAllFrames = -1;

constexpr int kDefaultScreenshotFrame   = 1;
constexpr int kDefaultMaxStepsPerformed = 0;
#ifdef ANGLE_STANDALONE_BENCHMARK
constexpr bool kStandaloneBenchmark = true;
#else
constexpr bool kStandaloneBenchmark = false;
#endif
inline bool OneFrame()
{
    return gStepsPerTrial == 1 || gMaxStepsPerformed == 1;
}
}  // namespace angle

#endif  // TESTS_PERF_TESTS_ANGLE_PERF_TEST_ARGS_H_
