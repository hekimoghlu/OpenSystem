/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#if USE(APPLE_INTERNAL_SDK)

#import <Foundation/NSURLFileTypeMappings.h>

#else

// FIXME: We should use UTI instead, but it's missing some mappings that this old SPI knows.
// Remove these methods once <rdar://problem/18042184> is fixed.
@interface NSURLFileTypeMappings
@end
@interface NSURLFileTypeMappings (Private)
+ (NSURLFileTypeMappings *)sharedMappings;
- (NSString *)MIMETypeForExtension:(NSString *)ext;
- (NSString *)preferredExtensionForMIMEType:(NSString *)type;
- (NSArray *)extensionsForMIMEType:(NSString *)type;
@end

#endif
