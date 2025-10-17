/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#ifndef _OTAAUTOASSETCLIENT_H_
#define _OTAAUTOASSETCLIENT_H_  1

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface OTAAutoAssetClient : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
- (nullable instancetype)initWithError:(NSError **)error;
- (nullable NSString *)startUsingLocalAsset;
- (void)stopUsingLocalAsset;
- (void)registerForAssetChangedNotificationsWithBlock:(void (^)(void))assetDidChangeHandler;
+ (BOOL)saveTrustStoreAssetPath:(NSString *)assetPath;
+ (nullable NSString *)validTrustStoreAssetPath:(NSString *)assetPath mustExist:(BOOL)mustExist;
+ (nullable NSString *)savedTrustStoreAssetPath;
@end

NS_ASSUME_NONNULL_END

__END_DECLS

#endif /* _OTAAUTOASSETCLIENT_H_ */
