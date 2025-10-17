/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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

//
//  SOSCircleDer.h
//  sec
//
//  Created by Richard Murphy on 1/22/15.
//
//

#ifndef _sec_SOSCircleDer_
#define _sec_SOSCircleDer_

#include <stdio.h>

size_t der_sizeof_data_or_null(CFDataRef data, CFErrorRef* error);

uint8_t* der_encode_data_or_null(CFDataRef data, CFErrorRef* error, const uint8_t* der, uint8_t* der_end);

const uint8_t* der_decode_data_or_null(CFAllocatorRef allocator, CFDataRef* data,
                                       CFErrorRef* error,
                                       const uint8_t* der, const uint8_t* der_end);

#endif /* defined(_sec_SOSCircleDer_) */
