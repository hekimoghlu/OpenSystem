/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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

#import <AppKit/NSSharingService_Private.h>

#else

#import <AppKit/NSSharingService.h>

typedef NS_ENUM(NSInteger, NSSharingServiceType) {
    NSSharingServiceTypeShare = 0,
    NSSharingServiceTypeViewer = 1,
    NSSharingServiceTypeEditor = 2
} NS_ENUM_AVAILABLE_MAC(10_10);

typedef NS_OPTIONS(NSUInteger, NSSharingServiceMask) {
    NSSharingServiceMaskShare = (1 << NSSharingServiceTypeShare),
    NSSharingServiceMaskViewer = (1 << NSSharingServiceTypeViewer),
    NSSharingServiceMaskEditor = (1 << NSSharingServiceTypeEditor),

    NSSharingServiceMaskAllTypes = 0xFFFF
} NS_ENUM_AVAILABLE_MAC(10_10);

@interface NSSharingService ()
+ (NSArray *)sharingServicesForItems:(NSArray *)items mask:(NSSharingServiceMask)maskForFiltering;
+ (void)getSharingServicesForItems:(NSArray *)items mask:(NSSharingServiceMask)maskForFiltering completion:(void(^)(NSArray *))completion;
@property (readonly) NSSharingServiceType type;
@property (readwrite, copy) NSString *name;
@end

#endif
