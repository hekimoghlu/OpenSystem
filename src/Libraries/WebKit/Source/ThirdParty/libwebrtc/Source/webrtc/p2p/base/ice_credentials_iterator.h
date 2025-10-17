/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
#ifndef P2P_BASE_ICE_CREDENTIALS_ITERATOR_H_
#define P2P_BASE_ICE_CREDENTIALS_ITERATOR_H_

#include <vector>

#include "p2p/base/transport_description.h"

namespace cricket {

class IceCredentialsIterator {
 public:
  explicit IceCredentialsIterator(const std::vector<IceParameters>&);
  virtual ~IceCredentialsIterator();

  // Get next pooled ice credentials.
  // Returns a new random credential if the pool is empty.
  IceParameters GetIceCredentials();

  static IceParameters CreateRandomIceCredentials();

 private:
  std::vector<IceParameters> pooled_ice_credentials_;
};

}  // namespace cricket

#endif  // P2P_BASE_ICE_CREDENTIALS_ITERATOR_H_
