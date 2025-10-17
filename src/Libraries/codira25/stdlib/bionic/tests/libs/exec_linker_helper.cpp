/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#include <stdio.h>

extern "C" const char* __progname;

const char* helper_func();

__attribute__((constructor))
static void ctor(int argc, char* argv[]) {
  printf("ctor: argc=%d argv[0]=%s\n", argc, argv[0]);
}

int main(int argc, char* argv[]) {
  printf("main: argc=%d argv[0]=%s\n", argc, argv[0]);
  printf("__progname=%s\n", __progname);
  printf("%s\n", helper_func());
  return 0;
}
