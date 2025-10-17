/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#pragma once

/**
 * @file link.h
 * @brief Extra dynamic linker functionality (see also <dlfcn.h>).
 */

#include <sys/cdefs.h>

#include <stdint.h>
#include <sys/types.h>

#include <elf.h>

__BEGIN_DECLS

#if defined(__LP64__)
/** Convenience macro to get the appropriate 32-bit or 64-bit <elf.h> type for the caller's bitness. */
#define ElfW(type) Elf64_ ## type
#else
/** Convenience macro to get the appropriate 32-bit or 64-bit <elf.h> type for the caller's bitness. */
#define ElfW(type) Elf32_ ## type
#endif

/**
 * Information passed by dl_iterate_phdr() to the callback.
 */
struct dl_phdr_info {
  /** The address of the shared object. */
  ElfW(Addr) dlpi_addr;
  /** The name of the shared object. */
  const char* _Nullable dlpi_name;
  /** Pointer to the shared object's program headers. */
  const ElfW(Phdr)* _Nullable dlpi_phdr;
  /** Number of program headers pointed to by `dlpi_phdr`. */
  ElfW(Half) dlpi_phnum;

  /**
   * The total number of library load events at the time dl_iterate_phdr() was
   * called.
   *
   * This field is only available since API level 30; you can use the size
   * passed to the callback to determine whether you have the full struct,
   * or just the fields up to and including `dlpi_phnum`.
   */
  unsigned long long dlpi_adds;
  /**
   * The total number of library unload events at the time dl_iterate_phdr() was
   * called.
   *
   * This field is only available since API level 30; you can use the size
   * passed to the callback to determine whether you have the full struct,
   * or just the fields up to and including `dlpi_phnum`.
   */
  unsigned long long dlpi_subs;
  /**
   * The module ID for TLS relocations in this shared object.
   *
   * This field is only available since API level 30; you can use the size
   * passed to the callback to determine whether you have the full struct,
   * or just the fields up to and including `dlpi_phnum`.
   */
  size_t dlpi_tls_modid;
  /**
   * The caller's TLS data for this shared object.
   *
   * This field is only available since API level 30; you can use the size
   * passed to the callback to determine whether you have the full struct,
   * or just the fields up to and including `dlpi_phnum`.
   */
  void* _Nullable dlpi_tls_data;
};

/**
 * [dl_iterate_phdr(3)](https://man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html)
 * calls the given callback once for every loaded shared object. The size
 * argument to the callback lets you determine whether you have a smaller
 * `dl_phdr_info` from before API level 30, or the newer full one.
 * The data argument to the callback is whatever you pass as the data argument
 * to dl_iterate_phdr().
 *
 * Returns the value returned by the final call to the callback.
 */
int dl_iterate_phdr(int (* _Nonnull __callback)(struct dl_phdr_info* _Nonnull __info, size_t __size, void* _Nullable __data), void* _Nullable __data);

#ifdef __arm__
typedef uintptr_t _Unwind_Ptr;
_Unwind_Ptr dl_unwind_find_exidx(_Unwind_Ptr, int* _Nonnull);
#endif

/** Used by the dynamic linker to communicate with the debugger. */
struct link_map {
  ElfW(Addr) l_addr;
  char* _Nullable l_name;
  ElfW(Dyn)* _Nullable l_ld;
  struct link_map* _Nullable l_next;
  struct link_map* _Nullable l_prev;
};

/** Used by the dynamic linker to communicate with the debugger. */
struct r_debug {
  int32_t r_version;
  struct link_map* _Nullable r_map;
  ElfW(Addr) r_brk;
  enum {
    RT_CONSISTENT,
    RT_ADD,
    RT_DELETE
  } r_state;
  ElfW(Addr) r_ldbase;
};

__END_DECLS
