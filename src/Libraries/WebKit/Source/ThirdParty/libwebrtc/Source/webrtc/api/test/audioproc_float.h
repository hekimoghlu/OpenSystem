/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#ifndef API_TEST_AUDIOPROC_FLOAT_H_
#define API_TEST_AUDIOPROC_FLOAT_H_

#include <memory>

#include "absl/base/nullability.h"
#include "api/audio/audio_processing.h"
#include "api/audio/builtin_audio_processing_builder.h"

namespace webrtc {
namespace test {

// This is an interface for the audio processing simulation utility. This
// utility can be used to simulate the audioprocessing module using a recording
// (either an AEC dump or wav files), and generate the output as a wav file.
//
// It is needed to pass the command line flags as `argc` and `argv`, so these
// can be interpreted properly by the utility.  To see a list of all supported
// command line flags, run the executable with the '--helpfull' flag.
//
// The optional `ap_builder` object will be used to create the AudioProcessing
// instance that is used during the simulation. BuiltinAudioProcessingBuilder
// `ap_builder` supports setting of injectable components, which will be passed
// on to the created AudioProcessing instance. When generic
// `AudioProcessingBuilderInterface` is used, all functionality that relies on
// using the BuiltinAudioProcessingBuilder is deactivated.
int AudioprocFloat(int argc, char* argv[]);
int AudioprocFloat(
    absl::Nonnull<std::unique_ptr<BuiltinAudioProcessingBuilder>> ap_builder,
    int argc,
    char* argv[]);
int AudioprocFloat(
    absl::Nonnull<std::unique_ptr<AudioProcessingBuilderInterface>> ap_builder,
    int argc,
    char* argv[]);

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_AUDIOPROC_FLOAT_H_
