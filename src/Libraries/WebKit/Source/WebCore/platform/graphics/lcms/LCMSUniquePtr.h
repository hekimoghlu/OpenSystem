/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#pragma once

#include <lcms2.h>
#include <memory>

namespace WebCore {

// Specializing unique_ptr for:
//   typedef void* cmsHPROFILE;
//   typedef void* cmsHTRANSFORM;

struct LCMSProfileDeleter {
    void operator() (cmsHPROFILE ptr) const
    {
        if (ptr)
            cmsCloseProfile(ptr);
    }
};
using LCMSProfilePtr = std::unique_ptr<void, LCMSProfileDeleter>;

struct LCMSTransformDeleter {
    void operator() (cmsHTRANSFORM ptr) const
    {
        if (ptr)
            cmsDeleteTransform(ptr);
    }
};
using LCMSTransformPtr = std::unique_ptr<void, LCMSTransformDeleter>;


} // namespace WebCore
