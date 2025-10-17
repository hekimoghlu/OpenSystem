/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#ifndef API_NETEQ_CUSTOM_NETEQ_FACTORY_H_
#define API_NETEQ_CUSTOM_NETEQ_FACTORY_H_

#include <memory>

#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/environment/environment.h"
#include "api/neteq/neteq.h"
#include "api/neteq/neteq_controller_factory.h"
#include "api/neteq/neteq_factory.h"
#include "api/scoped_refptr.h"

namespace webrtc {

// This factory can be used to generate NetEq instances that make use of a
// custom NetEqControllerFactory.
class CustomNetEqFactory : public NetEqFactory {
 public:
  explicit CustomNetEqFactory(
      std::unique_ptr<NetEqControllerFactory> controller_factory);
  ~CustomNetEqFactory() override;
  CustomNetEqFactory(const CustomNetEqFactory&) = delete;
  CustomNetEqFactory& operator=(const CustomNetEqFactory&) = delete;

  std::unique_ptr<NetEq> Create(
      const Environment& env,
      const NetEq::Config& config,
      scoped_refptr<AudioDecoderFactory> decoder_factory) const override;

 private:
  std::unique_ptr<NetEqControllerFactory> controller_factory_;
};

}  // namespace webrtc
#endif  // API_NETEQ_CUSTOM_NETEQ_FACTORY_H_
