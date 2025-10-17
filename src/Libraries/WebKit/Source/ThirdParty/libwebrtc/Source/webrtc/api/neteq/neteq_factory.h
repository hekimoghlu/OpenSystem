/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#ifndef API_NETEQ_NETEQ_FACTORY_H_
#define API_NETEQ_NETEQ_FACTORY_H_

#include <memory>

#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/environment/environment.h"
#include "api/neteq/neteq.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// Creates NetEq instances using the settings provided in the config struct.
class NetEqFactory {
 public:
  virtual ~NetEqFactory() = default;

  // Creates a new NetEq object, with parameters set in `config`. The `config`
  // object will only have to be valid for the duration of the call to this
  // method.
  virtual std::unique_ptr<NetEq> Create(
      const Environment& env,
      const NetEq::Config& config,
      scoped_refptr<AudioDecoderFactory> decoder_factory) const = 0;
};

}  // namespace webrtc
#endif  // API_NETEQ_NETEQ_FACTORY_H_
