/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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

@import Foundation;

#include "stdint.h"

void void_ptr_func(void * _Nonnull buffer);
void const_void_ptr_func(const void * _Nonnull buffer);
void opt_void_ptr_func(void * _Nullable buffer);
void const_opt_void_ptr_func(const void * _Nullable buffer);

void char_ptr_func(char *  _Nonnull buffer);
void const_char_ptr_func(const char *  _Nonnull buffer);

void opt_char_ptr_func(char * _Nullable buffer);
void const_opt_char_ptr_func(const char * _Nullable buffer);

void unsigned_char_ptr_func(unsigned char * _Nonnull buffer);
void const_unsigned_char_ptr_func(const unsigned char * _Nonnull buffer);

void opt_unsigned_char_ptr_func(char * _Nullable buffer);
void const_opt_unsigned_char_ptr_func(const char * _Nullable buffer);

void int_16_ptr_func(int16_t * _Nonnull buffer);
void int_32_ptr_func(int32_t * _Nonnull buffer);
void int_64_ptr_func(int64_t * _Nonnull buffer);

void opt_int_16_ptr_func(int16_t * _Nullable buffer);
void opt_int_32_ptr_func(int32_t * _Nullable buffer);
void opt_int_64_ptr_func(int64_t * _Nullable buffer);

void const_int_16_ptr_func(const int16_t * _Nonnull buffer);
void const_int_32_ptr_func(const int32_t * _Nonnull buffer);
void const_int_64_ptr_func(const int64_t * _Nonnull buffer);

void const_opt_int_16_ptr_func(const int16_t * _Nullable buffer);
void const_opt_int_32_ptr_func(const int32_t * _Nullable buffer);
void const_opt_int_64_ptr_func(const int64_t * _Nullable buffer);

void uint_16_ptr_func(uint16_t * _Nonnull buffer);
void uint_32_ptr_func(uint32_t * _Nonnull buffer);
void uint_64_ptr_func(uint64_t * _Nonnull buffer);

void opt_uint_16_ptr_func(uint16_t * _Nullable buffer);
void opt_uint_32_ptr_func(uint32_t * _Nullable buffer);
void opt_uint_64_ptr_func(uint64_t * _Nullable buffer);

void const_uint_16_ptr_func(const uint16_t * _Nonnull buffer);
void const_uint_32_ptr_func(const uint32_t * _Nonnull buffer);
void const_uint_64_ptr_func(const uint64_t * _Nonnull buffer);

void const_opt_uint_16_ptr_func(const uint16_t * _Nullable buffer);
void const_opt_uint_32_ptr_func(const uint32_t * _Nullable buffer);
void const_opt_uint_64_ptr_func(const uint64_t * _Nullable buffer);
