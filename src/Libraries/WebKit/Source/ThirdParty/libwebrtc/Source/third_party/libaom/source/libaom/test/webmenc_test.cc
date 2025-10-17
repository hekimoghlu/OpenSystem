/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#include <string>
#include "common/webmenc.h"
#include "gtest/gtest.h"

namespace {

#if CONFIG_WEBM_IO

class WebmencTest : public ::testing::Test {};

// All of these variations on output should be identical.
TEST(WebmencTest, ExtractEncoderSettingsOutput1) {
  const char *argv[] = { "aomenc", "-o", "output", "input",
                         "--target-bitrate=300" };
  int argc = 5;
  const std::string expected("version:1.2.3 --target-bitrate=300");
  char *result = extract_encoder_settings("1.2.3", argv, argc, "input");
  ASSERT_EQ(expected, std::string(result));
  free(result);
}

TEST(WebmencTest, ExtractEncoderSettingsOutput2) {
  const char *argv[] = { "aomenc", "--output", "bar", "foo", "--cpu-used=3" };
  int argc = 5;
  const std::string expected("version:abc --cpu-used=3");
  char *result = extract_encoder_settings("abc", argv, argc, "foo");
  ASSERT_EQ(expected, std::string(result));
  free(result);
}

TEST(WebmencTest, ExtractEncoderSettingsOutput3) {
  const char *argv[] = { "aomenc", "--cq-level=63", "--end-usage=q",
                         "--output=foo", "baz" };
  int argc = 5;
  const std::string expected("version:23 --cq-level=63 --end-usage=q");
  char *result = extract_encoder_settings("23", argv, argc, "baz");
  ASSERT_EQ(expected, std::string(result));
  free(result);
}

TEST(WebmencTest, ExtractEncoderSettingsInput) {
  // Check that input filename is filtered regardless of position.
  const char *argv[] = { "aomenc", "-o", "out", "input", "-p", "2" };
  int argc = 6;
  const char version[] = "1.0.0";
  const std::string expected("version:1.0.0 -p 2");
  char *result = extract_encoder_settings(version, argv, argc, "input");
  ASSERT_EQ(expected, std::string(result));
  free(result);

  const char *argv2[] = { "aomenc", "input", "-o", "out", "-p", "2" };
  result = extract_encoder_settings(version, argv2, argc, "input");
  ASSERT_EQ(expected, std::string(result));
  free(result);
}

#endif  // CONFIG_WEBM_IO
}  // namespace
