/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "modules/rtp_rtcp/source/rtp_header_extension_size.h"

#include "api/rtp_parameters.h"

namespace webrtc {

int RtpHeaderExtensionSize(rtc::ArrayView<const RtpExtensionSize> extensions,
                           const RtpHeaderExtensionMap& registered_extensions) {
  // RFC3550 Section 5.3.1
  static constexpr int kExtensionBlockHeaderSize = 4;

  int values_size = 0;
  int num_extensions = 0;
  int each_extension_header_size = 1;
  for (const RtpExtensionSize& extension : extensions) {
    int id = registered_extensions.GetId(extension.type);
    if (id == RtpHeaderExtensionMap::kInvalidId)
      continue;
    // All extensions should use same size header. Check if the `extension`
    // forces to switch to two byte header that allows larger id and value size.
    if (id > RtpExtension::kOneByteHeaderExtensionMaxId ||
        extension.value_size >
            RtpExtension::kOneByteHeaderExtensionMaxValueSize) {
      each_extension_header_size = 2;
    }
    values_size += extension.value_size;
    num_extensions++;
  }
  if (values_size == 0)
    return 0;
  int size = kExtensionBlockHeaderSize +
             each_extension_header_size * num_extensions + values_size;
  // Extension size specified in 32bit words,
  // so result must be multiple of 4 bytes. Round up.
  return size + 3 - (size + 3) % 4;
}

}  // namespace webrtc
