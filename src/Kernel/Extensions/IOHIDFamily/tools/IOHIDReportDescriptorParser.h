/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
//  IOHIDReportDescriptorParser.h
//  IOHIDFamily
//
//  Created by Rob Yepez on 2/23/13.
//
//

#ifndef IOHIDFamily_IOHIDReportParser_h
#define IOHIDFamily_IOHIDReportParser_h

#include <stdio.h>
#include <stdint.h>

extern void PrintHIDDescriptor(const uint8_t *reportDesc, uint32_t length);

#endif /* IOHIDFamily_IOHIDReportParser_h */
