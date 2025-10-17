/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#include <android/dlext.h>
#include <dlfcn.h>
#include <stdlib.h>

#include <string>

#include "../core_shared_libs.h"
#include "../dlext_private_tests.h"

extern "C" void global_function();
extern "C" void internal_function();

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s NS_PATH\n", argv[0]);
    fprintf(stderr, "NS_PATH   path to the ns_hidden_child_app directory\n");
    exit(1);
  }

  // Ensure that -Wl,--needed doesn't break the test by removing DT_NEEDED entries.
  global_function();
  internal_function();

  const char* app_lib_dir = argv[1];
  android_namespace_t* app_ns =
      android_create_namespace("app", nullptr, app_lib_dir, ANDROID_NAMESPACE_TYPE_ISOLATED,
                               nullptr, nullptr);
  if (app_ns == nullptr) {
    fprintf(stderr, "android_create_namespace failed: %s\n", dlerror());
    exit(1);
  }

  std::string public_libs = std::string(kCoreSharedLibs) + ":libns_hidden_child_public.so";
  if (!android_link_namespaces(app_ns, nullptr, public_libs.c_str())) {
    fprintf(stderr, "android_link_namespaces failed: %s\n", dlerror());
    exit(1);
  }

  android_dlextinfo ext = {
    .flags = ANDROID_DLEXT_USE_NAMESPACE,
    .library_namespace = app_ns,
  };
  void* app_lib = android_dlopen_ext("libns_hidden_child_app.so", RTLD_NOW | RTLD_LOCAL, &ext);
  if (app_lib == nullptr) {
    fprintf(stderr, "android_dlopen_ext failed: %s\n", dlerror());
    exit(1);
  }

  auto app_function = reinterpret_cast<void(*)()>(dlsym(app_lib, "app_function"));
  if (app_function == nullptr) {
    fprintf(stderr, "dlsym failed to find app_function: %s\n", dlerror());
    exit(1);
  }

  app_function();
  return 0;
}
