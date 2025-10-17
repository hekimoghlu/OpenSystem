/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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

#include <dlfcn.h>

// Work around for http://b/20049306, which isn't going to be fixed.
int defeat_sibling_call_optimization = 0;

extern "C" void* dlopen_b() {
  // This is supposed to succeed because this library has DT_RUNPATH
  // for libtest_dt_runpath_x.so which should be taken into account
  // by dlopen.
  void *handle = dlopen("libtest_dt_runpath_x.so", RTLD_NOW);
  if (handle != nullptr) {
    defeat_sibling_call_optimization++;
    return handle;
  }
  return nullptr;
}
