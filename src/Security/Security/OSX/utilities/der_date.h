/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#ifndef _UTILITIES_DER_DATE_H_
#define _UTILITIES_DER_DATE_H_

#include <CoreFoundation/CoreFoundation.h>

const uint8_t* der_decode_generalizedtime_body(CFAbsoluteTime *at, CFErrorRef *error,
                                               const uint8_t* der, const uint8_t *der_end);
const uint8_t* der_decode_universaltime_body(CFAbsoluteTime *at, CFErrorRef *error,
                                             const uint8_t* der, const uint8_t *der_end);

size_t der_sizeof_generalizedtime(CFAbsoluteTime at, CFErrorRef *error);
uint8_t* der_encode_generalizedtime(CFAbsoluteTime at, CFErrorRef *error,
                                    const uint8_t *der, uint8_t *der_end);

size_t der_sizeof_generalizedtime_body(CFAbsoluteTime at, CFErrorRef *error);
uint8_t* der_encode_generalizedtime_body(CFAbsoluteTime at, CFErrorRef *error,
                                         const uint8_t *der, uint8_t *der_end);
uint8_t* der_encode_generalizedtime_body_repair(CFAbsoluteTime at, CFErrorRef *error, bool repair,
                                                const uint8_t *der, uint8_t *der_end);

#endif /* _UTILITIES_DER_DATE_H_ */
