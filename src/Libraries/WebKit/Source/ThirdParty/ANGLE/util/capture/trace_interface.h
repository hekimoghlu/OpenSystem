/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
// Copyright 2023 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// trace_interface:
//   Interface shared between trace libraries and the test suite.
//

#ifndef UTIL_CAPTURE_TRACE_INTERFACE_H_
#define UTIL_CAPTURE_TRACE_INTERFACE_H_

#include <string>
#include <vector>

namespace angle
{

static constexpr size_t kTraceInfoMaxNameLen = 128;

enum class ReplayResourceMode
{
    Active,
    All,
};

struct TraceInfo
{
    char name[kTraceInfoMaxNameLen];
    bool initialized = false;
    uint32_t contextClientMajorVersion;
    uint32_t contextClientMinorVersion;
    uint32_t frameStart;
    uint32_t frameEnd;
    uint32_t drawSurfaceWidth;
    uint32_t drawSurfaceHeight;
    uint32_t drawSurfaceColorSpace;
    uint32_t displayPlatformType;
    uint32_t displayDeviceType;
    int configRedBits;
    int configBlueBits;
    int configGreenBits;
    int configAlphaBits;
    int configDepthBits;
    int configStencilBits;
    bool isBinaryDataCompressed;
    bool areClientArraysEnabled;
    bool isBindGeneratesResourcesEnabled;
    bool isWebGLCompatibilityEnabled;
    bool isRobustResourceInitEnabled;
    std::vector<std::string> traceFiles;
    int windowSurfaceContextId;
    std::vector<std::string> requiredExtensions;
    std::vector<int> keyFrames;
};

// Test suite calls into the trace library (fixture).
struct TraceFunctions
{
    virtual void SetupReplay()                    = 0;
    virtual void ReplayFrame(uint32_t frameIndex) = 0;
    virtual void ResetReplay()                    = 0;
    virtual void FinishReplay()                   = 0;

    virtual void SetBinaryDataDir(const char *dataDir)                        = 0;
    virtual void SetReplayResourceMode(const ReplayResourceMode resourceMode) = 0;
    virtual void SetTraceGzPath(const std::string &traceGzPath)               = 0;
    virtual void SetTraceInfo(const TraceInfo &traceInfo)                     = 0;

    virtual ~TraceFunctions() {}
};

// Trace library (fixture) calls into the test suite.
struct TraceCallbacks
{
    virtual uint8_t *LoadBinaryData(const char *fileName) = 0;

    virtual ~TraceCallbacks() {}
};

}  // namespace angle
#endif  // UTIL_CAPTURE_TRACE_INTERFACE_H_
