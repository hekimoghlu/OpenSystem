/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
#include "rtc_base/stream.h"

#include <string.h>

#include <cstdint>

#include "api/array_view.h"

namespace rtc {

///////////////////////////////////////////////////////////////////////////////
// StreamInterface
///////////////////////////////////////////////////////////////////////////////

StreamResult StreamInterface::WriteAll(ArrayView<const uint8_t> data,
                                       size_t& written,
                                       int& error) {
  StreamResult result = SR_SUCCESS;
  size_t total_written = 0, current_written;
  while (total_written < data.size()) {
    rtc::ArrayView<const uint8_t> this_slice =
        data.subview(total_written, data.size() - total_written);
    result = Write(this_slice, current_written, error);
    if (result != SR_SUCCESS)
      break;
    total_written += current_written;
  }
  written = total_written;
  return result;
}

bool StreamInterface::Flush() {
  return false;
}

StreamInterface::StreamInterface() {}

}  // namespace rtc
