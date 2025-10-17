/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

// Version.inc: Encapsulation of a GL version.

#include <tuple>

namespace gl
{

constexpr Version::Version()
    : Version(0, 0)
{
}

// Avoid conflicts with linux system defines
#undef major
#undef minor

constexpr Version::Version(unsigned int major_, unsigned int minor_)
    : major(major_),
      minor(minor_)
{
}

inline bool operator==(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) == std::tie(b.major, b.minor);
}

inline bool operator!=(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) != std::tie(b.major, b.minor);
}

inline bool operator>=(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) >= std::tie(b.major, b.minor);
}

inline bool operator<=(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) <= std::tie(b.major, b.minor);
}

inline bool operator<(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) < std::tie(b.major, b.minor);
}

inline bool operator>(const Version &a, const Version &b)
{
    return std::tie(a.major, a.minor) > std::tie(b.major, b.minor);
}

}  // namespace gl
