/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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

#if __has_include(<sys/hwprobe.h>)
#include <sys/hwprobe.h>
#include <sys/syscall.h>
#endif


#if defined(__riscv)
#include <riscv_vector.h>

__attribute__((noinline))
uint64_t scalar_cast(uint8_t const* p) {
  return *(uint64_t const*)p;
}

__attribute__((noinline))
uint64_t scalar_memcpy(uint8_t const* p) {
  uint64_t r;
  __builtin_memcpy(&r, p, sizeof(r));
  return r;
}

__attribute__((noinline))
uint64_t vector_memcpy(uint8_t* d, uint8_t const* p) {
  __builtin_memcpy(d, p, 16);
  return *(uint64_t const*)d;
}

__attribute__((noinline))
uint64_t vector_ldst(uint8_t* d, uint8_t const* p) {
  __riscv_vse8(d, __riscv_vle8_v_u8m1(p, 16), 16);
  return *(uint64_t const*)d;
}

__attribute__((noinline))
uint64_t vector_ldst64(uint8_t* d, uint8_t const* p) {
  __riscv_vse64((unsigned long *)d, __riscv_vle64_v_u64m1((const unsigned long *)p, 16), 16);
  return *(uint64_t const*)d;
}

// For testing scalar and vector unaligned accesses.
uint64_t tmp[3] = {1,1,1};
uint64_t dst[3] = {1,1,1};
#endif

TEST(sys_hwprobe, __riscv_hwprobe_misaligned_scalar) {
#if defined(__riscv)
  uint8_t* p = (uint8_t*)tmp + 1;
  ASSERT_NE(0U, scalar_cast(p));
  ASSERT_NE(0U, scalar_memcpy(p));
#else
  GTEST_SKIP() << "__riscv_hwprobe requires riscv64";
#endif
}

TEST(sys_hwprobe, __riscv_hwprobe_misaligned_vector) {
#if defined(__riscv)
  uint8_t* p = (uint8_t*)tmp + 1;
  uint8_t* d = (uint8_t*)dst + 1;

  ASSERT_NE(0U, vector_ldst(d, p));
  ASSERT_NE(0U, vector_memcpy(d, p));
  ASSERT_NE(0U, vector_ldst64(d, p));
#else
  GTEST_SKIP() << "__riscv_hwprobe requires riscv64";
#endif
}

TEST(sys_hwprobe, __riscv_hwprobe) {
#if defined(__riscv) && __has_include(<sys/hwprobe.h>)
  riscv_hwprobe probes[] = {{.key = RISCV_HWPROBE_KEY_IMA_EXT_0},
                            {.key = RISCV_HWPROBE_KEY_CPUPERF_0}};
  ASSERT_EQ(0, __riscv_hwprobe(probes, 2, 0, nullptr, 0));
  EXPECT_EQ(RISCV_HWPROBE_KEY_IMA_EXT_0, probes[0].key);
  EXPECT_TRUE((probes[0].value & RISCV_HWPROBE_IMA_FD) != 0);
  EXPECT_TRUE((probes[0].value & RISCV_HWPROBE_IMA_C) != 0);
  EXPECT_TRUE((probes[0].value & RISCV_HWPROBE_IMA_V) != 0);
  EXPECT_TRUE((probes[0].value & RISCV_HWPROBE_EXT_ZBA) != 0);
  EXPECT_TRUE((probes[0].value & RISCV_HWPROBE_EXT_ZBB) != 0);
  EXPECT_TRUE((probes[0].value & RISCV_HWPROBE_EXT_ZBS) != 0);

  EXPECT_EQ(RISCV_HWPROBE_KEY_CPUPERF_0, probes[1].key);
  EXPECT_TRUE((probes[1].value & RISCV_HWPROBE_MISALIGNED_MASK) == RISCV_HWPROBE_MISALIGNED_FAST);
#else
  GTEST_SKIP() << "__riscv_hwprobe requires riscv64";
#endif
}

TEST(sys_hwprobe, __riscv_hwprobe_syscall_vdso) {
#if defined(__riscv) && __has_include(<sys/hwprobe.h>)
  riscv_hwprobe probes_vdso[] = {{.key = RISCV_HWPROBE_KEY_IMA_EXT_0},
                                 {.key = RISCV_HWPROBE_KEY_CPUPERF_0}};
  ASSERT_EQ(0, __riscv_hwprobe(probes_vdso, 2, 0, nullptr, 0));

  riscv_hwprobe probes_syscall[] = {{.key = RISCV_HWPROBE_KEY_IMA_EXT_0},
                                    {.key = RISCV_HWPROBE_KEY_CPUPERF_0}};
  ASSERT_EQ(0, syscall(SYS_riscv_hwprobe, probes_syscall, 2, 0, nullptr, 0));

  // Check we got the same answers from the vdso and the syscall.
  EXPECT_EQ(RISCV_HWPROBE_KEY_IMA_EXT_0, probes_syscall[0].key);
  EXPECT_EQ(probes_vdso[0].key, probes_syscall[0].key);
  EXPECT_EQ(probes_vdso[0].value, probes_syscall[0].value);
  EXPECT_EQ(RISCV_HWPROBE_KEY_CPUPERF_0, probes_syscall[1].key);
  EXPECT_EQ(probes_vdso[1].key, probes_syscall[1].key);
  EXPECT_EQ(probes_vdso[1].value, probes_syscall[1].value);
#else
  GTEST_SKIP() << "__riscv_hwprobe requires riscv64";
#endif
}

TEST(sys_hwprobe, __riscv_hwprobe_fail) {
#if defined(__riscv) && __has_include(<sys/hwprobe.h>)
  riscv_hwprobe probes[] = {};
  ASSERT_EQ(EINVAL, __riscv_hwprobe(probes, 0, 0, nullptr, ~0));
#else
  GTEST_SKIP() << "__riscv_hwprobe requires riscv64";
#endif
}