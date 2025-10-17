/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#include "sdk/android/native_api/jni/java_types.h"

#include <memory>
#include <vector>

#include "sdk/android/generated_native_unittests_jni/JavaTypesTestHelper_jni.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {
namespace {
TEST(JavaTypesTest, TestJavaToNativeStringMap) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedJavaLocalRef<jobject> j_map =
      jni::Java_JavaTypesTestHelper_createTestStringMap(env);

  std::map<std::string, std::string> output = JavaToNativeStringMap(env, j_map);

  std::map<std::string, std::string> expected{
      {"one", "1"},
      {"two", "2"},
      {"three", "3"},
  };
  EXPECT_EQ(expected, output);
}

TEST(JavaTypesTest, TestNativeToJavaToNativeIntArray) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();

  std::vector<int32_t> test_data{1, 20, 300};

  ScopedJavaLocalRef<jintArray> array = NativeToJavaIntArray(env, test_data);
  EXPECT_EQ(test_data, JavaToNativeIntArray(env, array));
}

TEST(JavaTypesTest, TestNativeToJavaToNativeByteArray) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();

  std::vector<int8_t> test_data{1, 20, 30};

  ScopedJavaLocalRef<jbyteArray> array = NativeToJavaByteArray(env, test_data);
  EXPECT_EQ(test_data, JavaToNativeByteArray(env, array));
}

TEST(JavaTypesTest, TestNativeToJavaToNativeIntArrayLeakTest) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();

  std::vector<int32_t> test_data{1, 20, 300};

  for (int i = 0; i < 2000; i++) {
    ScopedJavaLocalRef<jintArray> array = NativeToJavaIntArray(env, test_data);
    EXPECT_EQ(test_data, JavaToNativeIntArray(env, array));
  }
}

TEST(JavaTypesTest, TestNativeToJavaToNativeByteArrayLeakTest) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();

  std::vector<int8_t> test_data{1, 20, 30};

  for (int i = 0; i < 2000; i++) {
    ScopedJavaLocalRef<jbyteArray> array =
        NativeToJavaByteArray(env, test_data);
    EXPECT_EQ(test_data, JavaToNativeByteArray(env, array));
  }
}
}  // namespace
}  // namespace test
}  // namespace webrtc
