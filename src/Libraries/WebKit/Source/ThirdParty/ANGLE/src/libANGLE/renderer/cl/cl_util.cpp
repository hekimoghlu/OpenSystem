/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// cl_utils.cpp: Helper functions for the CL back end

#include "libANGLE/renderer/cl/cl_util.h"

#include "libANGLE/Debug.h"

#include <cstdlib>

namespace rx
{

cl_version ExtractCLVersion(const std::string &version)
{
    const std::string::size_type spacePos = version.find(' ');
    const std::string::size_type dotPos   = version.find('.');
    if (spacePos == std::string::npos || dotPos == std::string::npos)
    {
        ERR() << "Failed to extract version from OpenCL version string: " << version;
        return 0u;
    }

    const long major = std::strtol(&version[spacePos + 1u], nullptr, 10);
    const long minor = std::strtol(&version[dotPos + 1u], nullptr, 10);
    if (major < 1 || major > 9 || minor < 0 || minor > 9)
    {
        ERR() << "Failed to extract version from OpenCL version string: " << version;
        return 0u;
    }
    return CL_MAKE_VERSION(static_cast<cl_uint>(major), static_cast<cl_uint>(minor), 0);
}

void RemoveUnsupportedCLExtensions(std::string &extensions)
{
    if (extensions.empty())
    {
        return;
    }
    using SizeT    = std::string::size_type;
    SizeT extStart = 0u;
    SizeT spacePos = extensions.find(' ');

    // Remove all unsupported extensions which are terminated by a space
    while (spacePos != std::string::npos)
    {
        const SizeT length = spacePos - extStart;
        if (IsCLExtensionSupported(extensions.substr(extStart, length)))
        {
            extStart = spacePos + 1u;
        }
        else
        {
            extensions.erase(extStart, length + 1u);
        }
        spacePos = extensions.find(' ', extStart);
    }

    // Remove last extension in string, if exists and unsupported
    if (extStart < extensions.length())
    {
        const SizeT length = extensions.length() - extStart;
        if (!IsCLExtensionSupported(extensions.substr(extStart, length)))
        {
            extensions.erase(extStart, length);
        }
    }

    // Remove trailing spaces
    while (!extensions.empty() && extensions.back() == ' ')
    {
        extensions.pop_back();
    }
}

void RemoveUnsupportedCLExtensions(NameVersionVector &extensions)
{
    auto extIt = extensions.cbegin();
    while (extIt != extensions.cend())
    {
        if (IsCLExtensionSupported(extIt->name))
        {
            ++extIt;
        }
        else
        {
            extIt = extensions.erase(extIt);
        }
    }
}

}  // namespace rx
