/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
#include <memory>

#include "sdk/android/generated_java_audio_jni/JavaAudioDeviceModule_jni.h"
#include "sdk/android/src/jni/audio_device/audio_record_jni.h"
#include "sdk/android/src/jni/audio_device/audio_track_jni.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

static jlong JNI_JavaAudioDeviceModule_CreateAudioDeviceModule(
    JNIEnv* env,
    const JavaParamRef<jobject>& j_context,
    const JavaParamRef<jobject>& j_audio_manager,
    const JavaParamRef<jobject>& j_webrtc_audio_record,
    const JavaParamRef<jobject>& j_webrtc_audio_track,
    int input_sample_rate,
    int output_sample_rate,
    jboolean j_use_stereo_input,
    jboolean j_use_stereo_output) {
  AudioParameters input_parameters;
  AudioParameters output_parameters;
  GetAudioParameters(env, j_context, j_audio_manager, input_sample_rate,
                     output_sample_rate, j_use_stereo_input,
                     j_use_stereo_output, &input_parameters,
                     &output_parameters);
  auto audio_input = std::make_unique<AudioRecordJni>(
      env, input_parameters, kHighLatencyModeDelayEstimateInMilliseconds,
      j_webrtc_audio_record);
  auto audio_output = std::make_unique<AudioTrackJni>(env, output_parameters,
                                                      j_webrtc_audio_track);
  return jlongFromPointer(CreateAudioDeviceModuleFromInputAndOutput(
                              AudioDeviceModule::kAndroidJavaAudio,
                              j_use_stereo_input, j_use_stereo_output,
                              kHighLatencyModeDelayEstimateInMilliseconds,
                              std::move(audio_input), std::move(audio_output))
                              .release());
}

}  // namespace jni
}  // namespace webrtc
