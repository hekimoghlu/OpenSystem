/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#include <stddef.h>

extern void* __dso_handle;

extern int __cxa_atexit(void (*)(void*), void*, void*);

__attribute__ ((visibility ("hidden")))
void __atexit_handler_wrapper(void* func) {
  if (func != NULL) {
    (*(void (*)(void))func)();
  }
}

__attribute__ ((visibility ("hidden")))
int atexit(void (*func)(void)) {
  return (__cxa_atexit(&__atexit_handler_wrapper, func, &__dso_handle));
}
