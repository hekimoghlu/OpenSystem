/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
#if PLATFORM(IOS) || PLATFORM(VISION)

#import <pal/spi/cocoa/LaunchServicesSPI.h>

#if USE(APPLE_INTERNAL_SDK)

// This space intentionally left blank

#else

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, LSInstallType) {
    LSInstallTypeIntentionalDowngrade = 8
};

@interface LSRecord : NSObject
@end

@interface LSBundleRecord : LSRecord
@property (readonly, nullable) NSString *bundleIdentifier;
@end

@interface LSApplicationRecord : LSBundleRecord
@property (readonly) BOOL placeholder;
@property (readonly) NSString *managementDomain;

- (instancetype)initWithBundleIdentifier:(NSString *)bundleIdentifier allowPlaceholder:(BOOL)allowPlaceholder error:(NSError **)outError;
@end

@interface LSEnumerator<__covariant ObjectType> : NSEnumerator<ObjectType>
@end

typedef NS_OPTIONS(uint64_t, LSApplicationEnumerationOptions) {
    LSApplicationEnumerationOptionsEnumeratePlaceholders = (1 << 6)
};

@interface LSApplicationRecord (Enumeration)
+ (LSEnumerator<LSApplicationRecord *> *)enumeratorWithOptions:(LSApplicationEnumerationOptions)options;
@end

@interface LSApplicationWorkspace : NSObject
+ (LSApplicationWorkspace *)defaultWorkspace;
- (void)openURL:(NSURL *)url configuration:(_LSOpenConfiguration *)configuration completionHandler:(void (^)(NSDictionary<NSString *, id> *result, NSError *error))completionHandler;
@end

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)
#endif // PLATFORM(IOS) || PLATFORM(VISION)
