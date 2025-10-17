/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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

// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#include "libxml/parser.h"
#include "libxml/tree.h"
#include "libxml/xmlversion.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

static void ignore (void * ctx, const char * msg, ...) {
  // Error handler to avoid spam of error messages from libxml parser.
}


// Entry point for LibFuzzer.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  xmlSetGenericErrorFunc(NULL, &ignore);

  std::vector<uint8_t> buffer(size + 1, 0);
  std::copy(data, data + size, buffer.data());

  xmlRegexpPtr x = xmlRegexpCompile(buffer.data());
  if (x)
    xmlRegFreeRegexp(x);

  return 0;
}
