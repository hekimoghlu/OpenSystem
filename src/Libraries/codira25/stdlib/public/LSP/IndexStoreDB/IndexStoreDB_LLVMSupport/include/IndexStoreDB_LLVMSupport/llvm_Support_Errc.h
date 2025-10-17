/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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

//===- toolchain/Support/Errc.h - Defines the toolchain::errc enum --------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
// While std::error_code works OK on all platforms we use, there are some
// some problems with std::errc that can be avoided by using our own
// enumeration:
//
// * std::errc is a namespace in some implementations. That meas that ADL
//   doesn't work and it is sometimes necessary to write std::make_error_code
//   or in templates:
//   using std::make_error_code;
//   make_error_code(...);
//
//   with this enum it is safe to always just use make_error_code.
//
// * Some implementations define fewer names than others. This header has
//   the intersection of all the ones we support.
//
// * std::errc is just marked with is_error_condition_enum. This means that
//   common patters like AnErrorCode == errc::no_such_file_or_directory take
//   4 virtual calls instead of two comparisons.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ERRC_H
#define LLVM_SUPPORT_ERRC_H

#include <IndexStoreDB_LLVMSupport/toolchain_Config_indexstoredb-prefix.h>

#include <system_error>

namespace toolchain {
enum class errc {
  argument_list_too_long = int(std::errc::argument_list_too_long),
  argument_out_of_domain = int(std::errc::argument_out_of_domain),
  bad_address = int(std::errc::bad_address),
  bad_file_descriptor = int(std::errc::bad_file_descriptor),
  broken_pipe = int(std::errc::broken_pipe),
  device_or_resource_busy = int(std::errc::device_or_resource_busy),
  directory_not_empty = int(std::errc::directory_not_empty),
  executable_format_error = int(std::errc::executable_format_error),
  file_exists = int(std::errc::file_exists),
  file_too_large = int(std::errc::file_too_large),
  filename_too_long = int(std::errc::filename_too_long),
  function_not_supported = int(std::errc::function_not_supported),
  illegal_byte_sequence = int(std::errc::illegal_byte_sequence),
  inappropriate_io_control_operation =
      int(std::errc::inappropriate_io_control_operation),
  interrupted = int(std::errc::interrupted),
  invalid_argument = int(std::errc::invalid_argument),
  invalid_seek = int(std::errc::invalid_seek),
  io_error = int(std::errc::io_error),
  is_a_directory = int(std::errc::is_a_directory),
  no_child_process = int(std::errc::no_child_process),
  no_lock_available = int(std::errc::no_lock_available),
  no_space_on_device = int(std::errc::no_space_on_device),
  no_such_device_or_address = int(std::errc::no_such_device_or_address),
  no_such_device = int(std::errc::no_such_device),
  no_such_file_or_directory = int(std::errc::no_such_file_or_directory),
  no_such_process = int(std::errc::no_such_process),
  not_a_directory = int(std::errc::not_a_directory),
  not_enough_memory = int(std::errc::not_enough_memory),
  not_supported = int(std::errc::not_supported),
  operation_not_permitted = int(std::errc::operation_not_permitted),
  permission_denied = int(std::errc::permission_denied),
  read_only_file_system = int(std::errc::read_only_file_system),
  resource_deadlock_would_occur = int(std::errc::resource_deadlock_would_occur),
  resource_unavailable_try_again =
      int(std::errc::resource_unavailable_try_again),
  result_out_of_range = int(std::errc::result_out_of_range),
  too_many_files_open_in_system = int(std::errc::too_many_files_open_in_system),
  too_many_files_open = int(std::errc::too_many_files_open),
  too_many_links = int(std::errc::too_many_links)
};

inline std::error_code make_error_code(errc E) {
  return std::error_code(static_cast<int>(E), std::generic_category());
}
}

namespace std {
template <> struct is_error_code_enum<toolchain::errc> : std::true_type {};
}
#endif
