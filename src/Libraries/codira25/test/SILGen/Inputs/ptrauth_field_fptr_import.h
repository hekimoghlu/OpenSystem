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

#ifndef TEST_C_FUNCTION
#define TEST_C_FUNCTION

struct SecureStruct {
  int (*__ptrauth(1, 0, 88)(secure_func_ptr1))();
  int (*__ptrauth(1, 0, 66)(secure_func_ptr2))();
};

struct AddressDiscriminatedSecureStruct {
  int (*__ptrauth(1, 1, 88)(secure_func_ptr1))();
  int (*__ptrauth(1, 1, 66)(secure_func_ptr2))();
};

struct SecureStruct *ptr_to_secure_struct;
struct AddressDiscriminatedSecureStruct
    *ptr_to_addr_discriminated_secure_struct;
#endif
