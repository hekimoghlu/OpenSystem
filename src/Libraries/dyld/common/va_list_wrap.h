/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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
#ifndef va_list_wrap_h
#define va_list_wrap_h

#include <stdarg.h>
#include <type_traits>

// a type-safe va_list wrapper
struct va_list_wrap
{
  // on x86_64 va_list is defined as a compiler synthesized type __va_list_tag[1]
  // when passed around functions that array becomes a pointer
  // so use std::decay to strip the array type and use a pointer instead
  std::decay_t<va_list> list;

  va_list_wrap(va_list list): list(list) {}
};

#endif /* mach_o_Version64_h */
