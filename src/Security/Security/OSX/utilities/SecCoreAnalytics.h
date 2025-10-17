/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#include <CoreFoundation/CoreFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif

void SecCoreAnalyticsSendValue(CFStringRef _Nonnull eventName, int64_t value);
void SecCoreAnalyticsSendKernEntropyAnalytics(void);
void SecCoreAnalyticsSendLegacyKeychainUIEvent(const char* _Nonnull dialogType, const char* _Nonnull clientPath);

#ifdef __cplusplus
}
#endif

#if __OBJC__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString* const SecCoreAnalyticsValue;

@interface SecCoreAnalytics : NSObject

+ (void)sendEvent:(NSString*) eventName event:(NSDictionary<NSString*,NSObject*> *)event;
+ (void)sendEventLazy:(NSString*) eventName builder:(NSDictionary<NSString*,NSObject*>* (^)(void))builder;

@end

NS_ASSUME_NONNULL_END

#endif /* __OBCJ__ */
