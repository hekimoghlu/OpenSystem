/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

//===-------- ScanFixture.h - Dependency scanning tests -*- C++ ---------*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "language-c/DependencyScan/DependencyScan.h"
#include "language/DependencyScan/DependencyScanningTool.h"
#include "gtest/gtest.h"
#include <string>

namespace language {
namespace unittest {

class ScanTest : public ::testing::Test {
public:
  ScanTest();
  ~ScanTest();

protected:
  // The tool used to execute tests' scanning queries
  language::dependencies::DependencyScanningTool ScannerTool;

  // Test workspace directory
  toolchain::SmallString<256> TemporaryTestWorkspace;

  // Path to where the Codira standard library can be found
  toolchain::SmallString<128> StdLibDir;
};

} // end namespace unittest
} // end namespace language
