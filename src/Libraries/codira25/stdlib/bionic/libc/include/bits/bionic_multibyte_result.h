/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

/**
 * @file bits/bionic_multibyte_result.h
 * @brief Named values for the magic number return values of multibyte
 * conversion APIs defined by C.
 */

#include <sys/cdefs.h>

#include <stddef.h>

__BEGIN_DECLS

/**
 * @brief The error values defined by C for multibyte conversion APIs.
 *
 * Refer to C23 7.30.1 Restartable multibyte/wide character conversion functions
 * for more details.
 */
enum : size_t {
  /// @brief An encoding error occurred. The bytes read are not a valid unicode
  /// character, nor are they a partially valid character.
  BIONIC_MULTIBYTE_RESULT_ILLEGAL_SEQUENCE = -1UL,
#define BIONIC_MULTIBYTE_RESULT_ILLEGAL_SEQUENCE BIONIC_MULTIBYTE_RESULT_ILLEGAL_SEQUENCE

  /// @brief The bytes read may produce a valid unicode character, but the
  /// sequence is incomplete. Future calls may complete the character.
  BIONIC_MULTIBYTE_RESULT_INCOMPLETE_SEQUENCE = -2UL,
#define BIONIC_MULTIBYTE_RESULT_INCOMPLETE_SEQUENCE BIONIC_MULTIBYTE_RESULT_INCOMPLETE_SEQUENCE

  /// @brief The output of the call was the result of a previous successful
  /// decoding. No new bytes were consumed.
  ///
  /// The common case for this return value is when mbrtoc16 returns the low
  /// surrogate of a pair.
  BIONIC_MULTIBYTE_RESULT_NO_BYTES_CONSUMED = -3UL,
#define BIONIC_MULTIBYTE_RESULT_NO_BYTES_CONSUMED BIONIC_MULTIBYTE_RESULT_NO_BYTES_CONSUMED
};

__END_DECLS
