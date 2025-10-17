/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#ifndef _SECCORECRYPTO_H_
#define _SECCORECRYPTO_H_

#include <ctype.h>
#include <stddef.h>

#include <CoreFoundation/CoreFoundation.h>

__BEGIN_DECLS

#ifndef CCDER_BOOL_SUPPORT
const uint8_t* ccder_decode_bool(bool* boolean, const uint8_t* der, const uint8_t *der_end);
size_t ccder_sizeof_bool(bool value __unused, CFErrorRef *error);
uint8_t* ccder_encode_bool(bool value, const uint8_t *der, uint8_t *der_end);
#endif

__END_DECLS

#endif /* _SECCORECRYPTO_H_ */
