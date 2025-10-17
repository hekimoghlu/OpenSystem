/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
//  SecBuffer.c
//  utilities
//
//  Created by Mitch Adler on 3/6/15.
//  Copyright Â© 2015 Apple Inc. All rights reserved.
//

#include <utilities/SecBuffer.h>

#include <strings.h>

#define stackBufferSizeLimit 2048

void PerformWithBuffer(size_t size, void (^operation)(size_t size, uint8_t *buffer)) {
    if (size == 0) {
        operation(0, NULL);
    } else if (size <= stackBufferSizeLimit) {
        uint8_t buffer[size];
        operation(size, buffer);
    } else {
        uint8_t *buffer = malloc(size);
        
        operation(size, buffer);
        
        if (buffer)
            free(buffer);
    }
}

void PerformWithBufferAndClear(size_t size, void (^operation)(size_t size, uint8_t *buffer)) {
    PerformWithBuffer(size, ^(size_t buf_size, uint8_t *buffer) {
        operation(buf_size, buffer);
        
        bzero(buffer, buf_size);
    });
}
