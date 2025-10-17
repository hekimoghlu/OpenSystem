/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
// Unit tests for Program and related classes.
//

#include <gtest/gtest.h>

#include "libANGLE/Program.h"

using namespace gl;

namespace
{

// Tests that the log length properly counts the terminating \0.
TEST(InfoLogTest, LogLengthCountsTerminator)
{
    InfoLog infoLog;
    EXPECT_EQ(0u, infoLog.getLength());
    infoLog << " ";

    // " \n\0" = 3 characters
    EXPECT_EQ(3u, infoLog.getLength());
}

// Tests that the log doesn't append newlines to an empty string
TEST(InfoLogTest, InfoLogEmptyString)
{
    InfoLog infoLog;
    EXPECT_EQ(0u, infoLog.getLength());
    infoLog << "";

    // "" = 3 characters
    EXPECT_EQ(0u, infoLog.getLength());
}

// Tests that newlines get appended to the info log properly.
TEST(InfoLogTest, AppendingNewline)
{
    InfoLog infoLog;

    infoLog << "First" << 1 << 'x';
    infoLog << "Second" << 2 << 'y';

    std::string expected = "First1x\nSecond2y\n";

    EXPECT_EQ(expected, infoLog.str());
}

}  // namespace
