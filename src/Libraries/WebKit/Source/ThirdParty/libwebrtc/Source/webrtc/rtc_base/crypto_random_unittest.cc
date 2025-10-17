/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
#include "rtc_base/crypto_random.h"

#include <string.h>

#include <cstring>
#include <string>
#include <utility>

#include "rtc_base/buffer.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace rtc {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Return;
using ::testing::WithArg;
using ::testing::WithArgs;

TEST(RandomTest, TestCreateRandomId) {
  CreateRandomId();
}

TEST(RandomTest, TestCreateRandomDouble) {
  for (int i = 0; i < 100; ++i) {
    double r = CreateRandomDouble();
    EXPECT_GE(r, 0.0);
    EXPECT_LT(r, 1.0);
  }
}

TEST(RandomTest, TestCreateNonZeroRandomId) {
  EXPECT_NE(0U, CreateRandomNonZeroId());
}

TEST(RandomTest, TestCreateRandomString) {
  std::string random = CreateRandomString(256);
  EXPECT_EQ(256U, random.size());
  std::string random2;
  EXPECT_TRUE(CreateRandomString(256, &random2));
  EXPECT_NE(random, random2);
  EXPECT_EQ(256U, random2.size());
}

TEST(RandomTest, TestCreateRandomData) {
  static size_t kRandomDataLength = 32;
  std::string random1;
  std::string random2;
  EXPECT_TRUE(CreateRandomData(kRandomDataLength, &random1));
  EXPECT_EQ(kRandomDataLength, random1.size());
  EXPECT_TRUE(CreateRandomData(kRandomDataLength, &random2));
  EXPECT_EQ(kRandomDataLength, random2.size());
  EXPECT_NE(0, memcmp(random1.data(), random2.data(), kRandomDataLength));
}

TEST(RandomTest, TestCreateRandomStringEvenlyDivideTable) {
  static std::string kUnbiasedTable("01234567");
  std::string random;
  EXPECT_TRUE(CreateRandomString(256, kUnbiasedTable, &random));
  EXPECT_EQ(256U, random.size());

  static std::string kBiasedTable("0123456789");
  EXPECT_FALSE(CreateRandomString(256, kBiasedTable, &random));
  EXPECT_EQ(0U, random.size());
}

TEST(RandomTest, TestCreateRandomUuid) {
  std::string random = CreateRandomUuid();
  EXPECT_EQ(36U, random.size());
}

TEST(RandomTest, TestCreateRandomForTest) {
  // Make sure we get the output we expect.
  SetRandomTestMode(true);
  EXPECT_EQ(2154761789U, CreateRandomId());
  EXPECT_EQ("h0ISP4S5SJKH/9EY", CreateRandomString(16));
  EXPECT_EQ("41706e92-cdd3-46d9-a22d-8ff1737ffb11", CreateRandomUuid());
  static size_t kRandomDataLength = 32;
  std::string random;
  EXPECT_TRUE(CreateRandomData(kRandomDataLength, &random));
  EXPECT_EQ(kRandomDataLength, random.size());
  Buffer expected(
      "\xbd\x52\x2a\x4b\x97\x93\x2f\x1c"
      "\xc4\x72\xab\xa2\x88\x68\x3e\xcc"
      "\xa3\x8d\xaf\x13\x3b\xbc\x83\xbb"
      "\x16\xf1\xcf\x56\x0c\xf5\x4a\x8b",
      kRandomDataLength);
  EXPECT_EQ(0, memcmp(expected.data(), random.data(), kRandomDataLength));

  // Reset and make sure we get the same output.
  SetRandomTestMode(true);
  EXPECT_EQ(2154761789U, CreateRandomId());
  EXPECT_EQ("h0ISP4S5SJKH/9EY", CreateRandomString(16));
  EXPECT_EQ("41706e92-cdd3-46d9-a22d-8ff1737ffb11", CreateRandomUuid());
  EXPECT_TRUE(CreateRandomData(kRandomDataLength, &random));
  EXPECT_EQ(kRandomDataLength, random.size());
  EXPECT_EQ(0, memcmp(expected.data(), random.data(), kRandomDataLength));

  // Test different character sets.
  SetRandomTestMode(true);
  std::string str;
  EXPECT_TRUE(CreateRandomString(16, "a", &str));
  EXPECT_EQ("aaaaaaaaaaaaaaaa", str);
  EXPECT_TRUE(CreateRandomString(16, "abcd", &str));
  EXPECT_EQ("dbaaabdaccbcabbd", str);

  // Turn off test mode for other tests.
  SetRandomTestMode(false);
}

class MockRandomGenerator : public RandomGenerator {
 public:
  MOCK_METHOD(void, Die, ());
  ~MockRandomGenerator() override { Die(); }

  MOCK_METHOD(bool, Init, (const void* seed, size_t len), (override));
  MOCK_METHOD(bool, Generate, (void* buf, size_t len), (override));
};

TEST(RandomTest, TestSetRandomGenerator) {
  std::unique_ptr<MockRandomGenerator> will_move =
      std::make_unique<MockRandomGenerator>();
  MockRandomGenerator* generator = will_move.get();
  SetRandomGenerator(std::move(will_move));

  EXPECT_CALL(*generator, Init(_, sizeof(int))).WillOnce(Return(true));
  EXPECT_TRUE(InitRandom(5));

  std::string seed = "seed";
  EXPECT_CALL(*generator, Init(seed.data(), seed.size()))
      .WillOnce(Return(true));
  EXPECT_TRUE(InitRandom(seed.data(), seed.size()));

  uint32_t id = 4658;
  EXPECT_CALL(*generator, Generate(_, sizeof(uint32_t)))
      .WillOnce(DoAll(WithArg<0>(Invoke([&id](void* p) {
                        std::memcpy(p, &id, sizeof(uint32_t));
                      })),
                      Return(true)));
  EXPECT_EQ(CreateRandomId(), id);

  EXPECT_CALL(*generator, Generate)
      .WillOnce(DoAll(
          WithArgs<0, 1>([](void* p, size_t len) { std::memset(p, 0, len); }),
          Return(true)));
  EXPECT_THAT(CreateRandomUuid(), Not(IsEmpty()));

  // Set the default random generator, and expect that mock generator is
  // not used beyond this point.
  EXPECT_CALL(*generator, Die);
  EXPECT_CALL(*generator, Generate).Times(0);
  SetDefaultRandomGenerator();
  EXPECT_THAT(CreateRandomUuid(), Not(IsEmpty()));
}

}  // namespace
}  // namespace rtc
