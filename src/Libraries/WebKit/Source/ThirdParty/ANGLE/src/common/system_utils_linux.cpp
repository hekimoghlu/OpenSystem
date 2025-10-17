/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// system_utils_linux.cpp: Implementation of OS-specific functions for Linux

#include "common/debug.h"
#include "system_utils.h"

#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>

namespace angle
{
std::string GetExecutablePath()
{
    // We cannot use lstat to get the size of /proc/self/exe as it always returns 0
    // so we just use a big buffer and hope the path fits in it.
    char path[4096];

    ssize_t result = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (result < 0 || static_cast<size_t>(result) >= sizeof(path) - 1)
    {
        return "";
    }

    path[result] = '\0';
    return path;
}

std::string GetExecutableDirectory()
{
    std::string executablePath = GetExecutablePath();
    size_t lastPathSepLoc      = executablePath.find_last_of("/");
    return (lastPathSepLoc != std::string::npos) ? executablePath.substr(0, lastPathSepLoc) : "";
}

const char *GetSharedLibraryExtension()
{
    return "so";
}

double GetCurrentSystemTime()
{
    struct timespec currentTime;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    return currentTime.tv_sec + currentTime.tv_nsec * 1e-9;
}

void SetCurrentThreadName(const char *name)
{
    // There's a 15-character (16 including '\0') limit.  If the name is too big (and ERANGE is
    // returned), name will be ignored.
    ASSERT(strlen(name) < 16);
    pthread_setname_np(pthread_self(), name);
}
}  // namespace angle
