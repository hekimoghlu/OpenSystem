/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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

// Copyright 2019 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/base_paths.h"
#include "base/check.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "base/path_service.h"

// BoringSSL requires a GetTestData function to pick up test data files. By
// default, BoringSSL generates a source file with the data embedded, but this
// exceeds the limit for the _CheckForTooLargeFiles presubmit check.
std::string GetTestData(const char *path) {
  base::FilePath file_path;
  base::PathService::Get(base::DIR_SOURCE_ROOT, &file_path);
  file_path = file_path.AppendASCII("third_party/boringssl/src");
  file_path = file_path.AppendASCII(path);

  std::string result;
  CHECK(base::ReadFileToString(file_path, &result));
  return result;
}
