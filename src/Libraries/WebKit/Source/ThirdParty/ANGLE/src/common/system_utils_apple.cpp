/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// system_utils_apple.cpp: Implementation of OS-specific functions for Apple platforms

#include "system_utils.h"

#include <unistd.h>

#include <CoreServices/CoreServices.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <cstdlib>
#include <vector>

#include <array>

namespace angle
{
std::string GetExecutablePath()
{
    std::string result;

    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);

    std::vector<char> buffer;
    buffer.resize(size + 1);

    _NSGetExecutablePath(buffer.data(), &size);
    buffer[size] = '\0';

    if (!strrchr(buffer.data(), '/'))
    {
        return "";
    }
    return buffer.data();
}

std::string GetExecutableDirectory()
{
    std::string executablePath = GetExecutablePath();
    size_t lastPathSepLoc      = executablePath.find_last_of("/");
    return (lastPathSepLoc != std::string::npos) ? executablePath.substr(0, lastPathSepLoc) : "";
}

double GetCurrentSystemTime()
{
    mach_timebase_info_data_t timebaseInfo;
    mach_timebase_info(&timebaseInfo);

    double secondCoeff = timebaseInfo.numer * 1e-9 / timebaseInfo.denom;
    return secondCoeff * mach_absolute_time();
}

void SetCurrentThreadName(const char *name)
{
    pthread_setname_np(name);
}
}  // namespace angle
