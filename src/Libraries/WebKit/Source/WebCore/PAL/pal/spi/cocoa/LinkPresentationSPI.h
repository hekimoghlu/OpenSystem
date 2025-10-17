/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#pragma once

#if !PLATFORM(WATCHOS) || USE(APPLE_INTERNAL_SDK)
#import <LinkPresentation/LinkPresentation.h>
#else
#import <Foundation/Foundation.h>
@interface LPLinkMetadata : NSObject<NSCopying, NSSecureCoding>
@property (nonatomic, retain) NSURL *URL;
@property (nonatomic, copy) NSString *title;
@end

@interface LPMetadataProvider : NSObject
@end
#endif

#if USE(APPLE_INTERNAL_SDK)

#import <LinkPresentation/LPMetadata.h>
#import <LinkPresentation/LPMetadataProviderPrivate.h>

#else

@interface LPSpecializationMetadata : NSObject <NSSecureCoding, NSCopying>
@end

@interface LPFileMetadata : LPSpecializationMetadata
@property (nonatomic, copy) NSString *name;
@property (nonatomic, copy) NSString *type;
@property (nonatomic, assign) uint64_t size;
@end

@interface LPLinkMetadata ()

- (void)_setIncomplete:(BOOL)incomplete;

@property (nonatomic, copy) LPSpecializationMetadata *specialization;

@end

#if !PLATFORM(APPLETV)

@interface LPMetadataProvider ()
- (LPLinkMetadata *)_startFetchingMetadataForURL:(NSURL *)URL completionHandler:(void(^)(NSError *))completionHandler;
@end

#endif // !PLATFORM(APPLETV)

#endif // USE(APPLE_INTERNAL_SDK)
