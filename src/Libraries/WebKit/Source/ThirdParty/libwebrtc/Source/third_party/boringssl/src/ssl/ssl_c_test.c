/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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

#include <openssl/ssl.h>

int BORINGSSL_enum_c_type_test(void);

int BORINGSSL_enum_c_type_test(void) {
#if defined(__cplusplus)
#error "This is testing the behaviour of the C compiler."
#error "It's pointless to build it in C++ mode."
#endif

  // In C++, the enums in ssl.h are explicitly typed as ints to allow them to
  // be predeclared. This function confirms that the C compiler believes them
  // to be the same size as ints. They may differ in signedness, however.
  return sizeof(enum ssl_private_key_result_t) == sizeof(int);
}
