/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "modules/desktop_capture/win/desktop_capture_utils.h"

#include "rtc_base/strings/string_builder.h"

namespace webrtc {
namespace desktop_capture {
namespace utils {

// Generates a human-readable string from a COM error.
std::string ComErrorToString(const _com_error& error) {
  char buffer[1024];
  rtc::SimpleStringBuilder string_builder(buffer);
  // Use _bstr_t to simplify the wchar to char conversion for ErrorMessage().
  _bstr_t error_message(error.ErrorMessage());
  string_builder.AppendFormat("HRESULT: 0x%08X, Message: %s", error.Error(),
                              static_cast<const char*>(error_message));
  return string_builder.str();
}

}  // namespace utils
}  // namespace desktop_capture
}  // namespace webrtc
