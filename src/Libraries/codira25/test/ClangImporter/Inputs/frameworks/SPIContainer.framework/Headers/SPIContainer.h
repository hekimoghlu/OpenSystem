/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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

#import <Foundation/Foundation.h>

#ifdef SPI_AVAILABLE
#undef SPI_AVAILABLE
#define SPI_AVAILABLE API_AVAILABLE
#endif

#ifdef __SPI_AVAILABLE
#undef __SPI_AVAILABLE
#define __SPI_AVAILABLE API_AVAILABLE
#endif

SPI_AVAILABLE(macos(10.7))
@interface SPIInterface1
- (instancetype)init;
@end

__SPI_AVAILABLE(macos(10.7))
@interface SPIInterface2
- (instancetype)init;
@end

@interface SharedInterface
  + (NSInteger)foo SPI_AVAILABLE(macos(10.7));
@end
