/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
//  SOSGenCount.h
//  sec
//
//  Created by Richard Murphy on 1/29/15.
//
//

#ifndef _sec_SOSGenCount_
#define _sec_SOSGenCount_

#include <CoreFoundation/CoreFoundation.h>

typedef CFNumberRef SOSGenCountRef;

int64_t SOSGetGenerationSint(SOSGenCountRef gen);
SOSGenCountRef SOSGenerationCreate(void);
SOSGenCountRef SOSGenerationCreateWithValue(int64_t value);
SOSGenCountRef SOSGenerationIncrementAndCreate(SOSGenCountRef gen);
SOSGenCountRef SOSGenerationCopy(SOSGenCountRef gen);
bool SOSGenerationIsOlder(SOSGenCountRef current, SOSGenCountRef proposed);
SOSGenCountRef SOSGenerationCreateWithBaseline(SOSGenCountRef reference);

SOSGenCountRef SOSGenCountCreateFromDER(CFAllocatorRef allocator, CFErrorRef* error,
                                        const uint8_t** der_p, const uint8_t *der_end);
size_t SOSGenCountGetDEREncodedSize(SOSGenCountRef gencount, CFErrorRef *error);
uint8_t *SOSGenCountEncodeToDER(SOSGenCountRef gencount, CFErrorRef* error, const uint8_t* der, uint8_t* der_end);

void SOSGenerationCountWithDescription(SOSGenCountRef gen, void (^operation)(CFStringRef description));
CFStringRef SOSGenerationCountCopyDescription(SOSGenCountRef gen);


#endif /* defined(_sec_SOSGenCount_) */
