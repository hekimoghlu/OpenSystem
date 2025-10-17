/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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

// Â© 2019 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <algorithm>
#include <array>
#include <vector>

#include "fuzzer_utils.h"
#include "unicode/unistr.h"

// Taken from third_party/icu/source/data/mappings/convrtrs.txt file.
static const std::array<const char*, 45> kConverters = {
  {
    "UTF-8",
    "utf-16be",
    "utf-16le",
    "UTF-32",
    "UTF-32BE",
    "UTF-32LE",
    "ibm866-html",
    "iso-8859-2-html",
    "iso-8859-3-html",
    "iso-8859-4-html",
    "iso-8859-5-html",
    "iso-8859-6-html",
    "iso-8859-7-html",
    "iso-8859-8-html",
    "ISO-8859-8-I",
    "iso-8859-10-html",
    "iso-8859-13-html",
    "iso-8859-14-html",
    "iso-8859-15-html",
    "iso-8859-16-html",
    "koi8-r-html",
    "koi8-u-html",
    "macintosh-html",
    "windows-874-html",
    "windows-1250-html",
    "windows-1251-html",
    "windows-1252-html",
    "windows-1253-html",
    "windows-1254-html",
    "windows-1255-html",
    "windows-1256-html",
    "windows-1257-html",
    "windows-1258-html",
    "x-mac-cyrillic-html",
    "windows-936-2000",
    "gb18030",
    "big5-html",
    "euc-jp-html",
    "ISO_2022,locale=ja,version=0",
    "shift_jis-html",
    "euc-kr-html",
    "ISO-2022-KR",
    "ISO-2022-CN",
    "ISO-2022-CN-EXT",
    "HZ-GB-2312"
  }
};

IcuEnvironment* env = new IcuEnvironment();

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1) {
    return 0;
  }

  // First byte is used for random converter selection.
  uint8_t rnd = *data;
  data++;
  size--;

  std::unique_ptr<char[]> fuzzbuff(new char[size]);
  std::memcpy(fuzzbuff.get(), data, size);

  icu::UnicodeString str(fuzzbuff.get(), size,
                         kConverters[rnd % kConverters.size()]);

  return 0;
}
