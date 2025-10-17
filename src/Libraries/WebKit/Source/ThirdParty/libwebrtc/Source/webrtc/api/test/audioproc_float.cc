/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#include "api/test/audioproc_float.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "api/audio/audio_processing.h"
#include "api/audio/builtin_audio_processing_builder.h"
#include "modules/audio_processing/test/audioproc_float_impl.h"

namespace webrtc {
namespace test {

int AudioprocFloat(int argc, char* argv[]) {
  return AudioprocFloatImpl(std::make_unique<BuiltinAudioProcessingBuilder>(),
                            argc, argv);
}

int AudioprocFloat(
    absl::Nonnull<std::unique_ptr<BuiltinAudioProcessingBuilder>> ap_builder,
    int argc,
    char* argv[]) {
  return AudioprocFloatImpl(std::move(ap_builder), argc, argv);
}

int AudioprocFloat(
    absl::Nonnull<std::unique_ptr<AudioProcessingBuilderInterface>> ap_builder,
    int argc,
    char* argv[]) {
  return AudioprocFloatImpl(std::move(ap_builder), argc, argv);
}

}  // namespace test
}  // namespace webrtc
