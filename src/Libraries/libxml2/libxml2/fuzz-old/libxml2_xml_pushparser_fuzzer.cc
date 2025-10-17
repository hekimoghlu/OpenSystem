/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

// Copyright 2015 The Chromium Authors. All rights reserved.
// Copyright 2017 Apple Inc. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <functional>
#include <limits>
#include <string>

#include "libxml/parser.h"
#include "libxml/xmlsave.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

static void ignore (void* ctx, const char* msg, ...) {
  // Error handler to avoid spam of error messages from libxml parser.
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  xmlSetGenericErrorFunc(NULL, &ignore);

  assert(size < std::numeric_limits<int>::max());
  int intSize = static_cast<int>(size);

  // Test default empty options value and some random combination.
  std::string data_string(reinterpret_cast<const char*>(data), size);
  const std::size_t data_hash = std::hash<std::string>()(data_string);
  const int max_option_value = std::numeric_limits<int>::max();
  const int random_option_value = data_hash % max_option_value;
  const int options[] = {0, random_option_value};

  int initialChunkSize = std::min<int>(4, intSize);
  int chunkSize = std::max<int>(100, std::min<int>(data_hash % 1024, intSize));

  for (const auto option_value : options) {
    if (auto ctxt = xmlCreatePushParserCtxt(NULL, NULL, (const char *)data, initialChunkSize, "noname.xml")) {
      xmlCtxtUseOptions(ctxt, option_value | XML_PARSE_NONET);
      int cur = initialChunkSize;
      do {
        if (cur + chunkSize >= size) {
          xmlParseChunk(ctxt, (const char *)data + cur, intSize - cur, 1);
          break;
        } else {
          xmlParseChunk(ctxt, (const char *)data + cur, chunkSize, 0);
          cur += chunkSize;
        }
      } while (cur < size);
      auto doc = ctxt->myDoc;
      xmlFreeParserCtxt(ctxt);

      auto buf = xmlBufferCreate();
      assert(buf);
      auto ctxtSave = xmlSaveToBuffer(buf, NULL, 0);
      xmlSaveDoc(ctxtSave, doc);
      xmlSaveClose(ctxtSave);
      xmlFreeDoc(doc);
      xmlBufferFree(buf);
    }
  }

  return 0;
}
