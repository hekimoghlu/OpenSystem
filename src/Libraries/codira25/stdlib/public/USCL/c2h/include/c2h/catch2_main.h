/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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

#include <thrust/detail/config/device_system.h>

#include <iostream>

//! @file
//! This file includes a custom Catch2 main function. When CMake is configured to build each test as a separate
//! executable, this header is included into each test. On the other hand, when all the tests are compiled into a single
//! executable, this header is excluded from the tests and included into catch2_runner.cpp

#include <catch2/catch_session.hpp>

#ifdef C2H_CONFIG_MAIN
#  if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#    include <c2h/catch2_runner_helper.h>

#    ifndef C2H_EXCLUDE_CATCH2_HELPER_IMPL
#      include "catch2_runner_helper.inl"
#    endif // !C2H_EXCLUDE_CATCH2_HELPER_IMPL
#  endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

int main(int argc, char* argv[])
{
  Catch::Session session;

#  if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  int device_id{};

  // Build a new parser on top of Catch's
  using namespace Catch::Clara;
  auto cli = session.cli() | Opt(device_id, "device")["-d"]["--device"]("device id to use");
  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0)
  {
    return returnCode;
  }

  set_device(device_id);
#  endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  return session.run();
}
#endif // C2H_CONFIG_MAIN
