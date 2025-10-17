/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#if PLATFORM(IOS_FAMILY)

#if USE(APPLE_INTERNAL_SDK)

#import <FrontBoardServices/FBSDisplay.h>
#import <FrontBoardServices/FBSOpenApplicationService.h>

#else

@interface FBSDisplayConfiguration : NSObject
@property (nonatomic, copy, readonly) NSString *name;
@end

extern NSString *const FBSActivateForEventOptionTypeBackgroundContentFetching;
extern NSString *const FBSOpenApplicationOptionKeyActions;
extern NSString *const FBSOpenApplicationOptionKeyActivateForEvent;
extern NSString *const FBSOpenApplicationOptionKeyActivateSuspended;
extern NSString *const FBSOpenApplicationOptionKeyPayloadOptions;
extern NSString *const FBSOpenApplicationOptionKeyPayloadURL;

@interface FBSOpenApplicationOptions : NSObject <NSCopying>
+ (instancetype)optionsWithDictionary:(NSDictionary *)dictionary;
@end

@class BSProcessHandle;
typedef void(^FBSOpenApplicationCompletion)(BSProcessHandle *process, NSError *error);

@interface FBSOpenApplicationService : NSObject
- (void)openApplication:(NSString *)bundleID withOptions:(FBSOpenApplicationOptions *)options completion:(FBSOpenApplicationCompletion)completion;
@end

#endif // USE(APPLE_INTERNAL_SDK)

// Forward declare this for all SDKs to get the extern C linkage
WTF_EXTERN_C_BEGIN
extern FBSOpenApplicationService *SBSCreateOpenApplicationService(void);
WTF_EXTERN_C_END

#endif // PLATFORM(IOS_FAMILY)
