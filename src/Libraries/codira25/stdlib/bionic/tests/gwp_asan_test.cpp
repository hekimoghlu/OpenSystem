/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/file.h>
#include <string>

#if defined(__BIONIC__)

#include "android-base/file.h"
#include "android-base/silent_death_test.h"
#include "android-base/test_utils.h"
#include "gwp_asan/options.h"
#include "platform/bionic/malloc.h"
#include "sys/system_properties.h"
#include "utils.h"

using gwp_asan_integration_DeathTest = SilentDeathTest;

// basename is a mess, use gnu basename explicitly to avoid the need for string
// mutation.
extern "C" const char* __gnu_basename(const char* path);

// GWP-ASan tests can run much slower, especially when combined with HWASan.
// Triple the deadline to avoid flakes (b/238585984).
extern "C" bool GetInitialArgs(const char*** args, size_t* num_args) {
  static const char* initial_args[] = {"--deadline_threshold_ms=270000"};
  *args = initial_args;
  *num_args = 1;
  return true;
}

// This file implements "torture testing" under GWP-ASan, where we sample every
// single allocation. The upper limit for the number of GWP-ASan allocations in
// the torture mode is is generally 40,000, so that svelte devices don't
// explode, as this uses ~163MiB RAM (4KiB per live allocation).
TEST(gwp_asan_integration, malloc_tests_under_torture) {
  // Do not override HWASan with GWP ASan.
  SKIP_WITH_HWASAN;

  RunGwpAsanTest("malloc.*:-malloc.mallinfo*");
}

class SyspropRestorer {
 private:
  std::vector<std::pair<std::string, std::string>> props_to_restore_;
  // System properties are global for a device, so the tests that mutate the
  // GWP-ASan system properties must be run mutually exclusive. Because
  // bionic-unit-tests is run in an isolated gtest fashion (each test is run in
  // its own process), we have to use flocks to synchronise between tests.
  int flock_fd_;

 public:
  SyspropRestorer() {
    std::string path = testing::internal::GetArgvs()[0];
    flock_fd_ = open(path.c_str(), O_RDONLY);
    EXPECT_NE(flock_fd_, -1) << "failed to open self for a flock";
    EXPECT_NE(flock(flock_fd_, LOCK_EX), -1) << "failed to flock myself";

    const char* basename = __gnu_basename(path.c_str());
    std::vector<std::string> props = {
        std::string("libc.debug.gwp_asan.sample_rate.") + basename,
        std::string("libc.debug.gwp_asan.process_sampling.") + basename,
        std::string("libc.debug.gwp_asan.max_allocs.") + basename,
        "libc.debug.gwp_asan.sample_rate.system_default",
        "libc.debug.gwp_asan.sample_rate.app_default",
        "libc.debug.gwp_asan.process_sampling.system_default",
        "libc.debug.gwp_asan.process_sampling.app_default",
        "libc.debug.gwp_asan.max_allocs.system_default",
        "libc.debug.gwp_asan.max_allocs.app_default",
    };

    size_t base_props_size = props.size();
    for (size_t i = 0; i < base_props_size; ++i) {
      props.push_back("persist." + props[i]);
    }

    std::string reset_log;

    for (const std::string& prop : props) {
      std::string value = GetSysprop(prop);
      props_to_restore_.emplace_back(prop, value);
      if (!value.empty()) {
        __system_property_set(prop.c_str(), "");
      }
    }
  }

  ~SyspropRestorer() {
    for (const auto& kv : props_to_restore_) {
      if (kv.second != GetSysprop(kv.first)) {
        __system_property_set(kv.first.c_str(), kv.second.c_str());
      }
    }
    close(flock_fd_);
  }

  static std::string GetSysprop(const std::string& name) {
    std::string value;
    const prop_info* pi = __system_property_find(name.c_str());
    if (pi == nullptr) return value;
    __system_property_read_callback(
        pi,
        [](void* cookie, const char* /* name */, const char* value, uint32_t /* serial */) {
          std::string* v = static_cast<std::string*>(cookie);
          *v = value;
        },
        &value);
    return value;
  }
};

TEST_F(gwp_asan_integration_DeathTest, DISABLED_assert_gwp_asan_enabled) {
  std::string maps;
  EXPECT_TRUE(android::base::ReadFileToString("/proc/self/maps", &maps));
  EXPECT_TRUE(maps.find("GWP-ASan") != std::string::npos) << maps;

  volatile int* x = new int;
  delete x;
  EXPECT_DEATH({ *x = 7; }, "");
}

// A weaker version of the above tests, only checking that GWP-ASan is enabled
// for any pointer, not *our* pointer. This allows us to test the system_default
// sysprops without potentially OOM-ing other random processes:
// b/273904016#comment5
TEST(gwp_asan_integration, DISABLED_assert_gwp_asan_enabled_weaker) {
  std::string maps;
  EXPECT_TRUE(android::base::ReadFileToString("/proc/self/maps", &maps));
  EXPECT_TRUE(maps.find("GWP-ASan") != std::string::npos) << maps;
}

TEST(gwp_asan_integration, DISABLED_assert_gwp_asan_disabled) {
  std::string maps;
  EXPECT_TRUE(android::base::ReadFileToString("/proc/self/maps", &maps));
  EXPECT_TRUE(maps.find("GWP-ASan") == std::string::npos);
}

TEST(gwp_asan_integration, sysprops_program_specific) {
  // Do not override HWASan with GWP ASan.
  SKIP_WITH_HWASAN;

  SyspropRestorer restorer;

  std::string path = testing::internal::GetArgvs()[0];
  const char* basename = __gnu_basename(path.c_str());
  __system_property_set((std::string("libc.debug.gwp_asan.sample_rate.") + basename).c_str(), "1");
  __system_property_set((std::string("libc.debug.gwp_asan.process_sampling.") + basename).c_str(),
                        "1");
  __system_property_set((std::string("libc.debug.gwp_asan.max_allocs.") + basename).c_str(),
                        "40000");

  RunSubtestNoEnv("gwp_asan_integration_DeathTest.DISABLED_assert_gwp_asan_enabled");
}

TEST(gwp_asan_integration, sysprops_persist_program_specific) {
  // Do not override HWASan with GWP ASan.
  SKIP_WITH_HWASAN;

  SyspropRestorer restorer;

  std::string path = testing::internal::GetArgvs()[0];
  const char* basename = __gnu_basename(path.c_str());
  __system_property_set(
      (std::string("persist.libc.debug.gwp_asan.sample_rate.") + basename).c_str(), "1");
  __system_property_set(
      (std::string("persist.libc.debug.gwp_asan.process_sampling.") + basename).c_str(), "1");
  __system_property_set((std::string("persist.libc.debug.gwp_asan.max_allocs.") + basename).c_str(),
                        "40000");

  RunSubtestNoEnv("gwp_asan_integration_DeathTest.DISABLED_assert_gwp_asan_enabled");
}

TEST(gwp_asan_integration, sysprops_non_persist_overrides_persist) {
  // Do not override HWASan with GWP ASan.
  SKIP_WITH_HWASAN;

  SyspropRestorer restorer;

  __system_property_set("libc.debug.gwp_asan.sample_rate.system_default", "1");
  __system_property_set("libc.debug.gwp_asan.process_sampling.system_default", "1");
  // Note, any processes launched elsewhere on the system right now will have
  // GWP-ASan enabled. Make sure that we only use a single slot, otherwise we
  // could end up causing said badly-timed processes to use up to 163MiB extra
  // penalty that 40,000 allocs would cause. See b/273904016#comment5 for more
  // context.
  __system_property_set("libc.debug.gwp_asan.max_allocs.system_default", "1");

  __system_property_set("persist.libc.debug.gwp_asan.sample_rate.system_default", "0");
  __system_property_set("persist.libc.debug.gwp_asan.process_sampling.system_default", "0");
  __system_property_set("persist.libc.debug.gwp_asan.max_allocs.system_default", "0");

  RunSubtestNoEnv("gwp_asan_integration.DISABLED_assert_gwp_asan_enabled_weaker");
}

TEST(gwp_asan_integration, sysprops_program_specific_overrides_default) {
  // Do not override HWASan with GWP ASan.
  SKIP_WITH_HWASAN;

  SyspropRestorer restorer;

  std::string path = testing::internal::GetArgvs()[0];
  const char* basename = __gnu_basename(path.c_str());
  __system_property_set(
      (std::string("persist.libc.debug.gwp_asan.sample_rate.") + basename).c_str(), "1");
  __system_property_set(
      (std::string("persist.libc.debug.gwp_asan.process_sampling.") + basename).c_str(), "1");
  __system_property_set((std::string("persist.libc.debug.gwp_asan.max_allocs.") + basename).c_str(),
                        "40000");

  __system_property_set("libc.debug.gwp_asan.sample_rate.system_default", "0");
  __system_property_set("libc.debug.gwp_asan.process_sampling.system_default", "0");
  __system_property_set("libc.debug.gwp_asan.max_allocs.system_default", "0");

  RunSubtestNoEnv("gwp_asan_integration_DeathTest.DISABLED_assert_gwp_asan_enabled");
}

TEST(gwp_asan_integration, sysprops_can_disable) {
  // Do not override HWASan with GWP ASan.
  SKIP_WITH_HWASAN;

  SyspropRestorer restorer;

  __system_property_set("libc.debug.gwp_asan.sample_rate.system_default", "0");
  __system_property_set("libc.debug.gwp_asan.process_sampling.system_default", "0");
  __system_property_set("libc.debug.gwp_asan.max_allocs.system_default", "0");

  RunSubtestNoEnv("gwp_asan_integration.DISABLED_assert_gwp_asan_disabled");
}

TEST(gwp_asan_integration, env_overrides_sysprop) {
  // Do not override HWASan with GWP ASan.
  SKIP_WITH_HWASAN;

  SyspropRestorer restorer;

  __system_property_set("libc.debug.gwp_asan.sample_rate.system_default", "0");
  __system_property_set("libc.debug.gwp_asan.process_sampling.system_default", "0");
  __system_property_set("libc.debug.gwp_asan.max_allocs.system_default", "0");

  RunGwpAsanTest("gwp_asan_integration_DeathTest.DISABLED_assert_gwp_asan_enabled");
}

#endif  // defined(__BIONIC__)
