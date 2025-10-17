/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_DEBUG_DUMP_WRITER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_DEBUG_DUMP_WRITER_H_

#include <memory>

#include "modules/audio_coding/audio_network_adaptor/controller.h"
#include "modules/audio_coding/audio_network_adaptor/include/audio_network_adaptor.h"
#include "rtc_base/system/file_wrapper.h"

#if WEBRTC_ENABLE_PROTOBUF
#ifdef WEBRTC_ANDROID_PLATFORM_BUILD
#include "external/webrtc/webrtc/modules/audio_coding/audio_network_adaptor/config.pb.h"
#else
#include "modules/audio_coding/audio_network_adaptor/config.pb.h"
#endif
#endif

namespace webrtc {

class DebugDumpWriter {
 public:
  static std::unique_ptr<DebugDumpWriter> Create(FILE* file_handle);

  virtual ~DebugDumpWriter() = default;

  virtual void DumpEncoderRuntimeConfig(const AudioEncoderRuntimeConfig& config,
                                        int64_t timestamp) = 0;

  virtual void DumpNetworkMetrics(const Controller::NetworkMetrics& metrics,
                                  int64_t timestamp) = 0;

#if WEBRTC_ENABLE_PROTOBUF
  virtual void DumpControllerManagerConfig(
      const audio_network_adaptor::config::ControllerManager&
          controller_manager_config,
      int64_t timestamp) = 0;
#endif
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_DEBUG_DUMP_WRITER_H_
