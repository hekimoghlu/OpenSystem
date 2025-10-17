/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#ifndef SOURCEKITDFUNCTIONS_H
#define SOURCEKITDFUNCTIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define SWIFT_SENDABLE __attribute__((__language_attr__("@Sendable")))

// Avoid including <sourcekitd/sourcekitd.h> to make sure we don't call the
// functions directly. But we need the types to form the function pointers.
// These are supposed to stay stable across toolchains.

typedef void *sourcekitd_api_object_t;
typedef struct sourcekitd_api_uid_s *sourcekitd_api_uid_t;
typedef void *sourcekitd_api_response_t;
typedef const void *sourcekitd_api_request_handle_t;

typedef struct {
  uint64_t data[3];
} sourcekitd_api_variant_t;

typedef enum {
  SOURCEKITD_API_VARIANT_TYPE_NULL = 0,
  SOURCEKITD_API_VARIANT_TYPE_DICTIONARY = 1,
  SOURCEKITD_API_VARIANT_TYPE_ARRAY = 2,
  SOURCEKITD_API_VARIANT_TYPE_INT64 = 3,
  SOURCEKITD_API_VARIANT_TYPE_STRING = 4,
  SOURCEKITD_API_VARIANT_TYPE_UID = 5,
  SOURCEKITD_API_VARIANT_TYPE_BOOL = 6,
  SOURCEKITD_API_VARIANT_TYPE_DOUBLE = 7,
  SOURCEKITD_API_VARIANT_TYPE_DATA = 8,
} sourcekitd_api_variant_type_t;

typedef void *sourcekitd_api_variant_functions_t;

typedef sourcekitd_api_variant_type_t (*sourcekitd_api_variant_functions_get_type_t)(
  sourcekitd_api_variant_t obj
);

typedef size_t (*sourcekitd_api_variant_functions_array_get_count_t)(
  sourcekitd_api_variant_t array
);

typedef sourcekitd_api_variant_t (*sourcekitd_api_variant_functions_array_get_value_t)(
  sourcekitd_api_variant_t array,
  size_t index
);

typedef bool (*sourcekitd_api_variant_array_applier_f_t)(
  size_t index,
  sourcekitd_api_variant_t value,
  void *_Null_unspecified context
);

typedef bool (*sourcekitd_api_variant_dictionary_applier_f_t)(
  _Null_unspecified sourcekitd_api_uid_t key,
  sourcekitd_api_variant_t value,
  void *_Null_unspecified context
);

typedef bool (*sourcekitd_api_variant_functions_dictionary_apply_t)(
  sourcekitd_api_variant_t dict,
  _Null_unspecified sourcekitd_api_variant_dictionary_applier_f_t applier,
  void *_Null_unspecified context
);

typedef enum {
  SOURCEKITD_API_ERROR_CONNECTION_INTERRUPTED = 1,
  SOURCEKITD_API_ERROR_REQUEST_INVALID = 2,
  SOURCEKITD_API_ERROR_REQUEST_FAILED = 3,
  SOURCEKITD_API_ERROR_REQUEST_CANCELLED = 4
} sourcekitd_api_error_t;

typedef void (^sourcekitd_api_interrupted_connection_handler_t)(void);
typedef void (^sourcekitd_api_response_receiver_t)(
  sourcekitd_api_response_t _Nullable resp
);

typedef sourcekitd_api_uid_t _Nullable (^sourcekitd_api_uid_from_str_handler_t)(
  const char *_Nullable uidStr
);
typedef const char *_Nullable (^sourcekitd_api_str_from_uid_handler_t)(
  sourcekitd_api_uid_t _Nullable uid
);

typedef void *sourcekitd_api_plugin_initialize_params_t;

typedef struct {
  void (*_Nonnull initialize)(void);
  void (*_Nonnull shutdown)(void);

  void (*_Nullable register_plugin_path)(
    const char *_Nullable clientPlugin,
    const char *_Nullable servicePlugin
  );

  _Null_unspecified sourcekitd_api_uid_t (*_Nonnull uid_get_from_cstr)(
    const char *_Nonnull string
  );
  _Null_unspecified sourcekitd_api_uid_t (*_Nonnull uid_get_from_buf)(
    const char *_Nonnull buf,
    size_t length
  );
  size_t (*_Nonnull uid_get_length)(
    sourcekitd_api_uid_t _Nonnull obj
  );
  const char *_Null_unspecified (*_Nonnull uid_get_string_ptr)(
    sourcekitd_api_uid_t _Nonnull obj
  );
  _Null_unspecified sourcekitd_api_object_t (*_Nonnull request_retain)(
    sourcekitd_api_object_t _Nonnull object
  );
  void (*_Nonnull request_release)(
    sourcekitd_api_object_t _Nonnull object
  );
  _Null_unspecified sourcekitd_api_object_t (*_Nonnull request_dictionary_create)(
    const _Nullable sourcekitd_api_uid_t *_Nullable keys,
    const _Nullable sourcekitd_api_object_t *_Nullable values,
    size_t count
  );
  void (*_Nonnull request_dictionary_set_value)(
    sourcekitd_api_object_t _Nonnull dict,
    sourcekitd_api_uid_t _Nonnull key,
    sourcekitd_api_object_t _Nonnull value
  );
  void (*_Nonnull request_dictionary_set_string)(
    sourcekitd_api_object_t _Nonnull dict,
    sourcekitd_api_uid_t _Nonnull key,
    const char *_Nonnull string
  );
  void (*_Nonnull request_dictionary_set_stringbuf)(
    sourcekitd_api_object_t _Nonnull dict,
    sourcekitd_api_uid_t _Nonnull key,
    const char *_Nonnull buf,
    size_t length
  );
  void (*_Nonnull request_dictionary_set_int64)(
    sourcekitd_api_object_t _Nonnull dict,
    sourcekitd_api_uid_t _Nonnull key,
    int64_t val
  );
  void (*_Nonnull request_dictionary_set_uid)(
    sourcekitd_api_object_t _Nonnull dict,
    sourcekitd_api_uid_t _Nonnull key,
    sourcekitd_api_uid_t _Nonnull uid
  );
  _Null_unspecified sourcekitd_api_object_t (*_Nonnull request_array_create)(
    const _Nullable sourcekitd_api_object_t *_Nullable objects,
    size_t count
  );
  void (*_Nonnull request_array_set_value)(
    sourcekitd_api_object_t _Nonnull array,
    size_t index,
    sourcekitd_api_object_t _Nonnull value
  );
  void (*_Nonnull request_array_set_string)(
    sourcekitd_api_object_t _Nonnull array,
    size_t index,
    const char *_Nonnull string
  );
  void (*_Nonnull request_array_set_stringbuf)(
    sourcekitd_api_object_t _Nonnull array,
    size_t index,
    const char *_Nonnull buf,
    size_t length
  );
  void (*_Nonnull request_array_set_int64)(
    sourcekitd_api_object_t _Nonnull array,
    size_t index,
    int64_t val
  );
  void (*_Nonnull request_array_set_uid)(
    sourcekitd_api_object_t _Nonnull array,
    size_t index,
    sourcekitd_api_uid_t _Nonnull uid
  );
  _Null_unspecified sourcekitd_api_object_t (*_Nonnull request_int64_create)(
    int64_t val
  );
  _Null_unspecified sourcekitd_api_object_t (*_Nonnull request_string_create)(
    const char *_Nonnull string
  );
  _Null_unspecified sourcekitd_api_object_t (*_Nonnull request_uid_create)(
    sourcekitd_api_uid_t _Nonnull uid
  );
  _Null_unspecified sourcekitd_api_object_t (*_Nonnull request_create_from_yaml)(
    const char *_Nonnull yaml,
    char *_Nullable *_Nullable error
  );
  void (*_Nonnull request_description_dump)(
    sourcekitd_api_object_t _Nonnull obj
  );
  char *_Null_unspecified (*_Nonnull request_description_copy)(
    sourcekitd_api_object_t _Nonnull obj
  );
  void (*_Nonnull response_dispose)(
    sourcekitd_api_response_t _Nonnull obj
  );
  bool (*_Nonnull response_is_error)(
    sourcekitd_api_response_t _Nonnull obj
  );
  sourcekitd_api_error_t (*_Nonnull response_error_get_kind)(
    sourcekitd_api_response_t _Nonnull err
  );
  const char *_Null_unspecified (*_Nonnull response_error_get_description)(
    sourcekitd_api_response_t _Nonnull err
  );
  sourcekitd_api_variant_t (*_Nonnull response_get_value)(
    sourcekitd_api_response_t _Nonnull resp
  );
  sourcekitd_api_variant_type_t (*_Nonnull variant_get_type)(
    sourcekitd_api_variant_t obj
  );
  sourcekitd_api_variant_t (*_Nonnull variant_dictionary_get_value)(
    sourcekitd_api_variant_t dict,
    sourcekitd_api_uid_t _Nonnull key
  );
  const char *_Null_unspecified (*_Nonnull variant_dictionary_get_string)(
    sourcekitd_api_variant_t dict,
    sourcekitd_api_uid_t _Nonnull key
  );
  int64_t (*_Nonnull variant_dictionary_get_int64)(
    sourcekitd_api_variant_t dict,
    sourcekitd_api_uid_t _Nonnull key
  );
  bool (*_Nonnull variant_dictionary_get_bool)(
    sourcekitd_api_variant_t dict,
    sourcekitd_api_uid_t _Nonnull key
  );
  _Null_unspecified sourcekitd_api_uid_t (*_Nonnull variant_dictionary_get_uid)(
    sourcekitd_api_variant_t dict,
    sourcekitd_api_uid_t _Nonnull key
  );
  size_t (*_Nonnull variant_array_get_count)(
    sourcekitd_api_variant_t array
  );
  sourcekitd_api_variant_t (*_Nonnull variant_array_get_value)(
    sourcekitd_api_variant_t array,
    size_t index
  );
  const char *_Null_unspecified (*_Nonnull variant_array_get_string)(
    sourcekitd_api_variant_t array,
    size_t index
  );
  int64_t (*_Nonnull variant_array_get_int64)(
    sourcekitd_api_variant_t array,
    size_t index
  );
  bool (*_Nonnull variant_array_get_bool)(
    sourcekitd_api_variant_t array,
    size_t index
  );
  _Null_unspecified sourcekitd_api_uid_t (*_Nonnull variant_array_get_uid)(
    sourcekitd_api_variant_t array,
    size_t index
  );
  int64_t (*_Nonnull variant_int64_get_value)(
    sourcekitd_api_variant_t obj
  );
  bool (*_Nonnull variant_bool_get_value)(
    sourcekitd_api_variant_t obj
  );
  double (*_Nullable variant_double_get_value)(
    sourcekitd_api_variant_t obj
  );
  size_t (*_Nonnull variant_string_get_length)(
    sourcekitd_api_variant_t obj
  );
  const char *_Null_unspecified (*_Nonnull variant_string_get_ptr)(
    sourcekitd_api_variant_t obj
  );
  size_t (*_Nullable variant_data_get_size)(
    sourcekitd_api_variant_t obj
  );
  const void *_Null_unspecified (*_Nullable variant_data_get_ptr)(
    sourcekitd_api_variant_t obj
  );
  _Null_unspecified sourcekitd_api_uid_t (*_Nonnull variant_uid_get_value)(
    sourcekitd_api_variant_t obj
  );
  void (*_Nonnull response_description_dump)(
    sourcekitd_api_response_t _Nonnull resp
  );
  void (*_Nonnull response_description_dump_filedesc)(
    sourcekitd_api_response_t _Nonnull resp,
    int fd
  );
  char *_Null_unspecified (*_Nonnull response_description_copy)(
    sourcekitd_api_response_t _Nonnull resp
  );
  void (*_Nonnull variant_description_dump)(
    sourcekitd_api_variant_t obj
  );
  void (*_Nonnull variant_description_dump_filedesc)(
    sourcekitd_api_variant_t obj,
    int fd
  );
  char *_Null_unspecified (*_Nonnull variant_description_copy)(
    sourcekitd_api_variant_t obj
  );
  _Null_unspecified sourcekitd_api_response_t (*_Nonnull send_request_sync)(
    _Nonnull sourcekitd_api_object_t req
  );
  void (*_Nonnull send_request)(
    sourcekitd_api_object_t _Nonnull req,
    _Nullable sourcekitd_api_request_handle_t *_Nullable out_handle,
    _Nullable SWIFT_SENDABLE sourcekitd_api_response_receiver_t receiver
  );
  void (*_Nonnull cancel_request)(
    _Nullable sourcekitd_api_request_handle_t handle
  );
  void (*_Nonnull set_notification_handler)(
    _Nullable SWIFT_SENDABLE sourcekitd_api_response_receiver_t receiver
  );
  void (*_Nonnull set_uid_handlers)(
    _Nullable SWIFT_SENDABLE sourcekitd_api_uid_from_str_handler_t uid_from_str,
    _Nullable SWIFT_SENDABLE sourcekitd_api_str_from_uid_handler_t str_from_uid
  );
} sourcekitd_api_functions_t;

#endif
