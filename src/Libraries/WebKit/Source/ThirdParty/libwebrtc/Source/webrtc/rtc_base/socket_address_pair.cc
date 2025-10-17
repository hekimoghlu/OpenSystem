/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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
#include "rtc_base/socket_address_pair.h"

namespace rtc {

SocketAddressPair::SocketAddressPair(const SocketAddress& src,
                                     const SocketAddress& dest)
    : src_(src), dest_(dest) {}

bool SocketAddressPair::operator==(const SocketAddressPair& p) const {
  return (src_ == p.src_) && (dest_ == p.dest_);
}

bool SocketAddressPair::operator<(const SocketAddressPair& p) const {
  if (src_ < p.src_)
    return true;
  if (p.src_ < src_)
    return false;
  if (dest_ < p.dest_)
    return true;
  if (p.dest_ < dest_)
    return false;
  return false;
}

size_t SocketAddressPair::Hash() const {
  return src_.Hash() ^ dest_.Hash();
}

}  // namespace rtc
