/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#include "sdk/android/src/jni/pc/media_constraints.h"

#include <memory>

#include "sdk/android/generated_peerconnection_jni/MediaConstraints_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

namespace {

// Helper for translating a List<Pair<String, String>> to a Constraints.
MediaConstraints::Constraints PopulateConstraintsFromJavaPairList(
    JNIEnv* env,
    const JavaRef<jobject>& j_list) {
  MediaConstraints::Constraints constraints;
  for (const JavaRef<jobject>& entry : Iterable(env, j_list)) {
    constraints.emplace_back(
        JavaToStdString(env, Java_KeyValuePair_getKey(env, entry)),
        JavaToStdString(env, Java_KeyValuePair_getValue(env, entry)));
  }
  return constraints;
}

}  // namespace

// Copies all needed data so Java object is no longer needed at return.
std::unique_ptr<MediaConstraints> JavaToNativeMediaConstraints(
    JNIEnv* env,
    const JavaRef<jobject>& j_constraints) {
  return std::make_unique<MediaConstraints>(
      PopulateConstraintsFromJavaPairList(
          env, Java_MediaConstraints_getMandatory(env, j_constraints)),
      PopulateConstraintsFromJavaPairList(
          env, Java_MediaConstraints_getOptional(env, j_constraints)));
}

}  // namespace jni
}  // namespace webrtc
