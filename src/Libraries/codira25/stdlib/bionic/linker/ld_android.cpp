/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#include <sys/cdefs.h>

extern "C" void __internal_linker_error() {
  __builtin_trap();
}

__strong_alias(__loader_android_create_namespace, __internal_linker_error);
__strong_alias(__loader_android_dlopen_ext, __internal_linker_error);
__strong_alias(__loader_android_dlwarning, __internal_linker_error);
__strong_alias(__loader_android_get_application_target_sdk_version, __internal_linker_error);
__strong_alias(__loader_android_get_LD_LIBRARY_PATH, __internal_linker_error);
__strong_alias(__loader_android_get_exported_namespace, __internal_linker_error);
__strong_alias(__loader_android_init_anonymous_namespace, __internal_linker_error);
__strong_alias(__loader_android_link_namespaces, __internal_linker_error);
__strong_alias(__loader_android_link_namespaces_all_libs, __internal_linker_error);
__strong_alias(__loader_android_set_application_target_sdk_version, __internal_linker_error);
__strong_alias(__loader_android_update_LD_LIBRARY_PATH, __internal_linker_error);
__strong_alias(__loader_cfi_fail, __internal_linker_error);
__strong_alias(__loader_android_handle_signal, __internal_linker_error);
__strong_alias(__loader_dl_iterate_phdr, __internal_linker_error);
__strong_alias(__loader_dladdr, __internal_linker_error);
__strong_alias(__loader_dlclose, __internal_linker_error);
__strong_alias(__loader_dlerror, __internal_linker_error);
__strong_alias(__loader_dlopen, __internal_linker_error);
__strong_alias(__loader_dlsym, __internal_linker_error);
__strong_alias(__loader_dlvsym, __internal_linker_error);
__strong_alias(__loader_add_thread_local_dtor, __internal_linker_error);
__strong_alias(__loader_remove_thread_local_dtor, __internal_linker_error);
__strong_alias(__loader_shared_globals, __internal_linker_error);
__strong_alias(__loader_android_set_16kb_appcompat_mode, __internal_linker_error);
#if defined(__arm__)
__strong_alias(__loader_dl_unwind_find_exidx, __internal_linker_error);
#endif
__strong_alias(rtld_db_dlactivity, __internal_linker_error);
