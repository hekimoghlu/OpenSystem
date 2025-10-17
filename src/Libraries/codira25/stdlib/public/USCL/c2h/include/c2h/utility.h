/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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

#include <cstdlib>
#include <cstring>
#include <string>
#include <typeinfo>
#ifdef __GNUC__
#  include <cxxabi.h>
#endif // __GNUC__

namespace c2h
{

// TODO(bgruber): duplicated version of thrust/testing/unittest/system.h
inline std::string demangle(const char* name)
{
#if __GNUC__ && !_NVHPC_CUDA
  int status     = 0;
  char* realname = abi::__cxa_demangle(name, 0, 0, &status);
  std::string result(realname);
  std::free(realname);
  return result;
#else // __GNUC__ && !_NVHPC_CUDA
  return name;
#endif // __GNUC__ && !_NVHPC_CUDA
}

// TODO(bgruber): duplicated version of thrust/testing/unittest/util.h
template <typename T>
std::string type_name()
{
  return demangle(typeid(T).name());
}

} // namespace c2h
