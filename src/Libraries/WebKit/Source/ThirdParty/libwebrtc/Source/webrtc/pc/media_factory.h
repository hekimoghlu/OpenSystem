/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#ifndef PC_MEDIA_FACTORY_H_
#define PC_MEDIA_FACTORY_H_

#include <memory>

#include "api/environment/environment.h"
#include "call/call.h"
#include "call/call_config.h"
#include "media/base/media_engine.h"

namespace webrtc {

// PeerConnectionFactoryDependencies is forward declared because of circular
// dependency between MediaFactory and PeerConnectionFactoryDependencies:
// PeerConnectionFactoryDependencies keeps an instance of MediaFactory and thus
// needs to know how to destroy it.
// MediaFactory mentions PeerConnectionFactoryDependencies in api, but does not
// need its full definition.
struct PeerConnectionFactoryDependencies;

// Interface repsponsible for constructing media specific classes for
// PeerConnectionFactory and PeerConnection.
class MediaFactory {
 public:
  virtual ~MediaFactory() = default;

  virtual std::unique_ptr<Call> CreateCall(CallConfig config) = 0;
  virtual std::unique_ptr<cricket::MediaEngineInterface> CreateMediaEngine(
      const Environment& env,
      PeerConnectionFactoryDependencies& dependencies) = 0;
};

}  // namespace webrtc

#endif  // PC_MEDIA_FACTORY_H_
