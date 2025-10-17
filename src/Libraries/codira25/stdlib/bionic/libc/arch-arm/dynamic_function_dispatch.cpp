/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
#include <fcntl.h>
#include <private/bionic_ifuncs.h>
#include <sys/syscall.h>

extern "C" {

enum CpuVariant {
  kUnknown = 0,
  kGeneric,
  kCortexA7,
  kCortexA9,
  kCortexA53,
  kCortexA55,
  kKrait,
  kKryo,
};

static constexpr int MAX_CPU_NAME_LEN = 12;
struct CpuVariantNames {
  alignas(alignof(int)) char name[MAX_CPU_NAME_LEN];
  CpuVariant variant;
};

static constexpr CpuVariantNames cpu_variant_names[] = {
    {"cortex-a76", kCortexA55},
    {"kryo385", kCortexA55},
    {"cortex-a75", kCortexA55},
    {"kryo", kKryo},
    {"cortex-a73", kCortexA55},
    {"cortex-a55", kCortexA55},
    {"cortex-a53", kCortexA53},
    {"krait", kKrait},
    {"cortex-a9", kCortexA9},
    {"cortex-a7", kCortexA7},
    // kUnknown indicates the end of this array.
    {"", kUnknown},
};

static long ifunc_open(const char* pathname) {
  register long r0 __asm__("r0") = AT_FDCWD;
  register long r1 __asm__("r1") = reinterpret_cast<long>(pathname);
  register long r2 __asm__("r2") = O_RDONLY;
  register long r3 __asm__("r3") = 0;
  register long r7 __asm__("r7") = __NR_openat;
  __asm__ volatile("swi #0"
                   : "=r"(r0)
                   : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r7));
  return r0;
}

static ssize_t ifunc_read(int fd, void* buf, size_t count) {
  register long r0 __asm__("r0") = fd;
  register long r1 __asm__("r1") = reinterpret_cast<long>(buf);
  register long r2 __asm__("r2") = count;
  register long r7 __asm__("r7") = __NR_read;
  __asm__ volatile("swi #0"
                   : "=r"(r0)
                   : "r"(r0), "r"(r1), "r"(r2), "r"(r7)
                   : "memory");
  return r0;
}

static int ifunc_close(int fd) {
  register long r0 __asm__("r0") = fd;
  register long r7 __asm__("r7") = __NR_close;
  __asm__ volatile("swi #0" : "=r"(r0) : "r"(r0), "r"(r7));
  return r0;
}

static bool is_same_name(const char* a, const char* b) {
  static_assert(MAX_CPU_NAME_LEN % sizeof(int) == 0, "");
  const int* ia = reinterpret_cast<const int*>(a);
  const int* ib = reinterpret_cast<const int*>(b);
  for (size_t i = 0; i < MAX_CPU_NAME_LEN / sizeof(int); ++i) {
    if (ia[i] != ib[i]) {
      return false;
    }
  }
  return true;
}

static CpuVariant init_cpu_variant() {
  int fd = ifunc_open("/dev/cpu_variant:arm");
  if (fd < 0) return kGeneric;

  alignas(alignof(int)) char name[MAX_CPU_NAME_LEN] = {};

  int bytes_read, total_read = 0;
  while (total_read < MAX_CPU_NAME_LEN - 1 &&
         (bytes_read = ifunc_read(fd, name + total_read,
                                  MAX_CPU_NAME_LEN - 1 - total_read)) > 0) {
    total_read += bytes_read;
  }
  ifunc_close(fd);

  if (bytes_read != 0) {
    // The file is too big. We haven't reach the end. Or maybe there is an
    // error when reading.
    return kGeneric;
  }
  name[total_read] = 0;

  const CpuVariantNames* cpu_variant = cpu_variant_names;
  while (cpu_variant->variant != kUnknown) {
    if (is_same_name(cpu_variant->name, name)) {
      return cpu_variant->variant;
    }
    cpu_variant++;
  }
  return kGeneric;
}

static CpuVariant get_cpu_variant() {
  static CpuVariant cpu_variant = kUnknown;
  if (cpu_variant == kUnknown) {
    cpu_variant = init_cpu_variant();
  }
  return cpu_variant;
}

DEFINE_IFUNC_FOR(memmove) {
  RETURN_FUNC(memmove_func_t, memmove_a15);
}
MEMMOVE_SHIM()

DEFINE_IFUNC_FOR(memcpy) {
  return memmove_resolver(hwcap);
}
MEMCPY_SHIM()

// On arm32, __memcpy() is not publicly exposed, but gets called by memmove()
// in cases where the copy is known to be overlap-safe.
typedef void* __memcpy_func_t(void*, const void*, size_t);
DEFINE_IFUNC_FOR(__memcpy) {
  switch (get_cpu_variant()) {
    case kCortexA7:
      RETURN_FUNC(__memcpy_func_t, __memcpy_a7);
    case kCortexA9:
      RETURN_FUNC(__memcpy_func_t, __memcpy_a9);
    case kKrait:
      RETURN_FUNC(__memcpy_func_t, __memcpy_krait);
    case kCortexA53:
      RETURN_FUNC(__memcpy_func_t, __memcpy_a53);
    case kCortexA55:
      RETURN_FUNC(__memcpy_func_t, __memcpy_a55);
    case kKryo:
      RETURN_FUNC(__memcpy_func_t, __memcpy_kryo);
    default:
      RETURN_FUNC(__memcpy_func_t, __memcpy_a15);
  }
}
DEFINE_STATIC_SHIM(void* __memcpy(void* dst, const void* src, size_t n) {
  FORWARD(__memcpy)(dst, src, n);
})

DEFINE_IFUNC_FOR(__memset_chk) {
  switch (get_cpu_variant()) {
    case kCortexA7:
    case kCortexA53:
    case kCortexA55:
    case kKryo:
      RETURN_FUNC(__memset_chk_func_t, __memset_chk_a7);
    case kCortexA9:
      RETURN_FUNC(__memset_chk_func_t, __memset_chk_a9);
    case kKrait:
      RETURN_FUNC(__memset_chk_func_t, __memset_chk_krait);
    default:
      RETURN_FUNC(__memset_chk_func_t, __memset_chk_a15);
  }
}
__MEMSET_CHK_SHIM()

DEFINE_IFUNC_FOR(memset) {
  switch (get_cpu_variant()) {
    case kCortexA7:
    case kCortexA53:
    case kCortexA55:
    case kKryo:
      RETURN_FUNC(memset_func_t, memset_a7);
    case kCortexA9:
      RETURN_FUNC(memset_func_t, memset_a9);
    case kKrait:
      RETURN_FUNC(memset_func_t, memset_krait);
    default:
      RETURN_FUNC(memset_func_t, memset_a15);
  }
}
MEMSET_SHIM()

DEFINE_IFUNC_FOR(strcpy) {
  switch (get_cpu_variant()) {
    case kCortexA9:
      RETURN_FUNC(strcpy_func_t, strcpy_a9);
    default:
      RETURN_FUNC(strcpy_func_t, strcpy_a15);
  }
}
STRCPY_SHIM()

DEFINE_IFUNC_FOR(__strcpy_chk) {
  switch (get_cpu_variant()) {
    case kCortexA7:
      RETURN_FUNC(__strcpy_chk_func_t, __strcpy_chk_a7);
    case kCortexA9:
      RETURN_FUNC(__strcpy_chk_func_t, __strcpy_chk_a9);
    case kKrait:
    case kKryo:
      RETURN_FUNC(__strcpy_chk_func_t, __strcpy_chk_krait);
    case kCortexA53:
      RETURN_FUNC(__strcpy_chk_func_t, __strcpy_chk_a53);
    case kCortexA55:
      RETURN_FUNC(__strcpy_chk_func_t, __strcpy_chk_a55);
    default:
      RETURN_FUNC(__strcpy_chk_func_t, __strcpy_chk_a15);
  }
}
__STRCPY_CHK_SHIM()

DEFINE_IFUNC_FOR(stpcpy) {
  switch (get_cpu_variant()) {
    case kCortexA9:
      RETURN_FUNC(stpcpy_func_t, stpcpy_a9);
    default:
      RETURN_FUNC(stpcpy_func_t, stpcpy_a15);
  }
}
STPCPY_SHIM()

DEFINE_IFUNC_FOR(strcat) {
  switch (get_cpu_variant()) {
    case kCortexA9:
      RETURN_FUNC(strcat_func_t, strcat_a9);
    default:
      RETURN_FUNC(strcat_func_t, strcat_a15);
  }
}
STRCAT_SHIM()

DEFINE_IFUNC_FOR(__strcat_chk) {
  switch (get_cpu_variant()) {
    case kCortexA7:
      RETURN_FUNC(__strcat_chk_func_t, __strcat_chk_a7);
    case kCortexA9:
      RETURN_FUNC(__strcat_chk_func_t, __strcat_chk_a9);
    case kKrait:
    case kKryo:
      RETURN_FUNC(__strcat_chk_func_t, __strcat_chk_krait);
    case kCortexA53:
      RETURN_FUNC(__strcat_chk_func_t, __strcat_chk_a53);
    case kCortexA55:
      RETURN_FUNC(__strcat_chk_func_t, __strcat_chk_a55);
    default:
      RETURN_FUNC(__strcat_chk_func_t, __strcat_chk_a15);
  }
}
__STRCAT_CHK_SHIM()

DEFINE_IFUNC_FOR(strcmp) {
  RETURN_FUNC(strcmp_func_t, strcmp_a15);
}
STRCMP_SHIM()

DEFINE_IFUNC_FOR(strlen) {
  switch (get_cpu_variant()) {
    case kCortexA9:
      RETURN_FUNC(strlen_func_t, strlen_a9);
    default:
      RETURN_FUNC(strlen_func_t, strlen_a15);
  }
}
STRLEN_SHIM()

}  // extern "C"
