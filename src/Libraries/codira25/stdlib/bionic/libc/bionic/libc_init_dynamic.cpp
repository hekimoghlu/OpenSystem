/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
/*
 * This source files provides two important functions for dynamic
 * executables:
 *
 * - a C runtime initializer (__libc_preinit), which is called by
 *   the dynamic linker when libc.so is loaded. This happens before
 *   any other initializer (e.g. static C++ constructors in other
 *   shared libraries the program depends on).
 *
 * - a program launch function (__libc_init), which is called after
 *   all dynamic linking has been performed.
 */

#include <elf.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "bionic/pthread_internal.h"
#include "libc_init_common.h"

#include "async_safe/log.h"
#include "private/bionic_defs.h"
#include "private/bionic_elf_tls.h"
#include "private/bionic_globals.h"
#include "platform/bionic/macros.h"
#include "private/bionic_ssp.h"
#include "private/bionic_tls.h"
#include "private/KernelArgumentBlock.h"
#include "sys/system_properties.h"
#include "sysprop_helpers.h"

static bool starts_with(const char* s, const char* prefix) {
  return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool is_debuggable_build() {
  char pv[8];
  return get_property_value("ro.debuggable", pv, sizeof(pv)) && strcmp(pv, "1") == 0;
}

extern "C" const char* __gnu_basename(const char* path);

extern "C" {
  extern void netdClientInit(void);
  extern int __cxa_atexit(void (*)(void *), void *, void *);
};

void memtag_stack_dlopen_callback() {
  if (__pthread_internal_remap_stack_with_mte()) {
    async_safe_format_log(ANDROID_LOG_DEBUG, "libc", "remapped stacks as PROT_MTE");
  }
}

// Use an initializer so __libc_sysinfo will have a fallback implementation
// while .preinit_array constructors run.
#if defined(__i386__)
__LIBC_HIDDEN__ void* __libc_sysinfo = reinterpret_cast<void*>(__libc_int0x80);
#endif

extern "C" __attribute__((weak)) void __hwasan_library_loaded(ElfW(Addr) base,
                                                              const ElfW(Phdr)* phdr,
                                                              ElfW(Half) phnum);
extern "C" __attribute__((weak)) void __hwasan_library_unloaded(ElfW(Addr) base,
                                                                const ElfW(Phdr)* phdr,
                                                                ElfW(Half) phnum);

static void init_prog_id(libc_globals* globals) {
  char exe_path[500];
  ssize_t readlink_res = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1 /* space for NUL terminator */);
  if (readlink_res <= 0) {
    return;
  }
  exe_path[readlink_res] = '\0';

  int prog_id = 0;
  int flags = 0;

#define IS(prog) (!strcmp(exe_path, prog))

  if (IS("/apex/com.google.pixel.camera.hal/bin/hw/android.hardware.camera.provider@2.7-service-google")) {
    prog_id = PROG_PIXEL_CAMERA_PROVIDER_SERVICE;
  }
  else if (IS("/system/bin/surfaceflinger")) {
    prog_id = PROG_SURFACEFLINGER;
  }
  else if (IS("/vendor/bin/hw/android.hardware.audio.service")) {
    // needed for Pixel Tablet as of Android 15, see https://github.com/GrapheneOS/os-issue-tracker/issues/4306
    flags = GLOBAL_FLAG_DISABLE_HARDENED_MALLOC;
  } else if (IS("/vendor/bin/shared_modem_platform")) {
    char device_name[PROP_VALUE_MAX];
    get_property_value("ro.product.name", device_name, sizeof(device_name));
    if (strcmp(device_name, "tokay") == 0
        || strcmp(device_name, "komodo") == 0
        || strcmp(device_name, "comet") == 0
        || strcmp(device_name, "caiman") == 0
    ) {
      flags = GLOBAL_FLAG_DISABLE_HARDENED_MALLOC;
    }
  }

#undef IS

  bool is_debuggable = is_debuggable_build();
  const bool is_vendor_prog = starts_with(exe_path, "/vendor/") || starts_with(exe_path, "/apex/com.google.");
  if (is_debuggable && is_vendor_prog) {
    const char* basename = __gnu_basename(exe_path);
    static const char propName[] = "persist.device_config.memory_safety_native.hardened_malloc.mode_override.process.";
    char sysprop_name[512];
    char sysprop_value[PROP_VALUE_MAX];
    async_safe_format_buffer(sysprop_name, sizeof(sysprop_name), "%s%s", propName,
                             basename);
    get_property_value(sysprop_name, sysprop_value, sizeof(sysprop_value));
    if (strcmp("disabled", sysprop_value) == 0) {
      flags = GLOBAL_FLAG_DISABLE_HARDENED_MALLOC;
    } else if (strcmp("enabled", sysprop_value) == 0) {
      prog_id = 0;
      flags = 0;
    }
  }

  // libc_globals struct is write-protected
  globals->flags = flags;
  globals->prog_id = prog_id;
}

int get_prog_id() {
  return __libc_globals->prog_id;
}

// We need a helper function for __libc_preinit because compiling with LTO may
// inline functions requiring a stack protector check, but __stack_chk_guard is
// not initialized at the start of __libc_preinit. __libc_preinit_impl will run
// after __stack_chk_guard is initialized and therefore can safely have a stack
// protector.
__attribute__((noinline))
static void __libc_preinit_impl() {
#if defined(__i386__)
  __libc_init_sysinfo();
#endif

  // Register libc.so's copy of the TLS generation variable so the linker can
  // update it when it loads or unloads a shared object.
  TlsModules& tls_modules = __libc_shared_globals()->tls_modules;
  tls_modules.generation_libc_so = &__libc_tls_generation_copy;
  __libc_tls_generation_copy = tls_modules.generation;

  __libc_init_globals();
  __libc_init_common();
  __libc_init_scudo();

#if __has_feature(hwaddress_sanitizer)
  // Notify the HWASan runtime library whenever a library is loaded or unloaded
  // so that it can update its shadow memory.
  // This has to happen before _libc_init_malloc which might dlopen to load
  // profiler libraries.
  __libc_shared_globals()->load_hook = __hwasan_library_loaded;
  __libc_shared_globals()->unload_hook = __hwasan_library_unloaded;
#endif

  // Hooks for various libraries to let them know that we're starting up.
  __libc_globals.mutate([](libc_globals* globals) {
    init_prog_id(globals);
    __libc_init_malloc(globals);

    // save the default SIGABRT handler to support restoring it with mallopt(M_BIONIC_RESTORE_DEFAULT_SIGABRT_HANDLER)
    sigaction(SIGABRT, nullptr, &globals->saved_sigabrt_handler);
  });

  // Install reserved signal handlers for assisting the platform's profilers.
  __libc_init_profiling_handlers();

  __libc_init_fork_handler();

  __libc_shared_globals()->set_target_sdk_version_hook = __libc_set_target_sdk_version;

  netdClientInit();
}

// We flag the __libc_preinit function as a constructor to ensure that
// its address is listed in libc.so's .init_array section.
// This ensures that the function is called by the dynamic linker as
// soon as the shared library is loaded.
// We give this constructor priority 1 because we want libc's constructor
// to run before any others (such as the jemalloc constructor), and lower
// is better (http://b/68046352).
__attribute__((constructor(1))) static void __libc_preinit() {
  // The linker has initialized its copy of the global stack_chk_guard, and filled in the main
  // thread's TLS slot with that value. Initialize the local global stack guard with its value.
  __stack_chk_guard = reinterpret_cast<uintptr_t>(__get_tls()[TLS_SLOT_STACK_GUARD]);

  __libc_preinit_impl();
}

// This function is called from the executable's _start entry point
// (see arch-$ARCH/bionic/crtbegin.c), which is itself called by the dynamic
// linker after it has loaded all shared libraries the executable depends on.
//
// Note that the dynamic linker has also run all constructors in the
// executable at this point.
__noreturn void __libc_init(void* raw_args,
                            void (*onexit)(void) __unused,
                            int (*slingshot)(int, char**, char**),
                            structors_array_t const * const structors) {
  BIONIC_STOP_UNWIND;

  KernelArgumentBlock args(raw_args);

  // Several Linux ABIs don't pass the onexit pointer, and the ones that
  // do never use it.  Therefore, we ignore it.

  // The executable may have its own destructors listed in its .fini_array
  // so we need to ensure that these are called when the program exits
  // normally.
  if (structors->fini_array) {
    __cxa_atexit(__libc_fini,structors->fini_array,nullptr);
  }

  __libc_init_mte_late();

  // This roundabout way is needed so we don't use the static libc linked into the linker, which
  // will not affect the process.
  __libc_shared_globals()->memtag_stack_dlopen_callback = memtag_stack_dlopen_callback;

  exit(slingshot(args.argc - __libc_shared_globals()->initial_linker_arg_count,
                 args.argv + __libc_shared_globals()->initial_linker_arg_count,
                 args.envp));
}

extern "C" libc_shared_globals* __loader_shared_globals();

__LIBC_HIDDEN__ libc_shared_globals* __libc_shared_globals() {
  return __loader_shared_globals();
}
