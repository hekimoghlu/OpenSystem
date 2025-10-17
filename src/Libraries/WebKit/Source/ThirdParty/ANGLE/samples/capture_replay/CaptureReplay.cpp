/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
// CaptureReplay: Template for replaying a frame capture with ANGLE.

#include "SampleApplication.h"

#include <functional>

#include "util/capture/frame_capture_test_utils.h"

class CaptureReplaySample : public SampleApplication
{
  public:
    CaptureReplaySample(int argc, char **argv, const angle::TraceInfo &traceInfo)
        : SampleApplication("CaptureReplaySample",
                            argc,
                            argv,
                            ClientType::ES3_0,
                            traceInfo.drawSurfaceWidth,
                            traceInfo.drawSurfaceHeight),
          mTraceInfo(traceInfo)
    {}

    bool initialize() override
    {
        mTraceLibrary.reset(new angle::TraceLibrary("capture_replay_sample_trace"));
        assert(mTraceLibrary->valid());

        std::stringstream binaryPathStream;
        binaryPathStream << angle::GetExecutableDirectory() << angle::GetPathSeparator()
                         << ANGLE_CAPTURE_REPLAY_SAMPLE_DATA_DIR;
        mTraceLibrary->setBinaryDataDir(binaryPathStream.str().c_str());
        mTraceLibrary->setupReplay();
        return true;
    }

    void destroy() override { mTraceLibrary->finishReplay(); }

    void draw() override
    {
        // Compute the current frame, looping from frameStart to frameEnd.
        uint32_t frame = mTraceInfo.frameStart +
                         (mCurrentFrame % ((mTraceInfo.frameEnd - mTraceInfo.frameStart) + 1));
        if (mPreviousFrame > frame)
        {
            mTraceLibrary->resetReplay();
        }
        mTraceLibrary->replayFrame(frame);
        mPreviousFrame = frame;
        mCurrentFrame++;
    }

  private:
    uint32_t mCurrentFrame  = 0;
    uint32_t mPreviousFrame = 0;
    const angle::TraceInfo mTraceInfo;
    std::unique_ptr<angle::TraceLibrary> mTraceLibrary;
};

int main(int argc, char **argv)
{
    std::string exeDir = angle::GetExecutableDirectory();

    std::stringstream traceJsonPathStream;
    traceJsonPathStream << exeDir << angle::GetPathSeparator()
                        << ANGLE_CAPTURE_REPLAY_SAMPLE_DATA_DIR << angle::GetPathSeparator()
                        << "angle_capture.json";

    std::string traceJsonPath = traceJsonPathStream.str();

    angle::TraceInfo traceInfo = {};
    if (!angle::LoadTraceInfoFromJSON("capture_replay_sample_trace", traceJsonPath, &traceInfo))
    {
        std::cout << "Unable to load trace data: " << traceJsonPath << "\n";
        return 1;
    }

    CaptureReplaySample app(argc, argv, traceInfo);
    return app.run();
}
