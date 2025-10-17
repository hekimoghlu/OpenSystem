/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#import <CoreGraphics/CoreGraphics.h>

#if TARGET_OS_IPHONE
#import <WebCore/AbstractPasteboard.h>
#endif

#if TARGET_OS_IOS || (defined(TARGET_OS_VISION) && TARGET_OS_VISION)

@protocol UIDropSession;

typedef NS_ENUM(NSInteger, WebPreferredPresentationStyle) {
    WebPreferredPresentationStyleUnspecified,
    WebPreferredPresentationStyleInline,
    WebPreferredPresentationStyleAttachment
};

NS_ASSUME_NONNULL_BEGIN

@interface NSItemProvider (WebCoreExtras)
@property (nonatomic, readonly) BOOL web_containsFileURLAndFileUploadContent;
@property (nonatomic, readonly) NSArray<NSString *> *web_fileUploadContentTypes;
@end

/*! A WebItemProviderRegistrar encapsulates a single call to register something to an item provider.
 @discussion Classes that implement this protocol each represent a different way of writing data to
 an item provider. Some examples include setting a chunk of data corresponding to a type identifier,
 or registering a NSItemProviderWriting-conformant object, or registering a type to a promised file
 where the data has been written.
 */
@protocol WebItemProviderRegistrar <NSObject>
- (void)registerItemProvider:(NSItemProvider *)itemProvider;

@optional
@property (nonatomic, readonly) id <NSItemProviderWriting> representingObjectForClient;
@property (nonatomic, readonly) NSString *typeIdentifierForClient;
@property (nonatomic, readonly) NSData *dataForClient;
@end

typedef void(^WebItemProviderFileCallback)(NSURL * _Nullable, NSError * _Nullable);

/*! A WebItemProviderRegistrationInfoList represents a series of registration calls used to set up a
 single item provider.
 @discussion The order of items specified in the list (lowest indices first) is the order in which
 objects or data are registered to the item provider, and therefore indicates the relative fidelity
 of each item. Private UTI types, such as those vended through the injected editing bundle SPI, are
 considered to be higher fidelity than the other default types.
 */
WEBCORE_EXPORT @interface WebItemProviderRegistrationInfoList : NSObject

- (void)addRepresentingObject:(id <NSItemProviderWriting>)object;
- (void)addData:(NSData *)data forType:(NSString *)typeIdentifier;
- (void)addPromisedType:(NSString *)typeIdentifier fileCallback:(void(^)(WebItemProviderFileCallback))callback;

@property (nonatomic) CGSize preferredPresentationSize;
@property (nonatomic, copy) NSString *suggestedName;
@property (nonatomic, readonly, nullable) __kindof NSItemProvider *itemProvider;

@property (nonatomic) WebPreferredPresentationStyle preferredPresentationStyle;
@property (nonatomic, copy) NSData *teamData;

- (NSUInteger)numberOfItems;
- (nullable id <WebItemProviderRegistrar>)itemAtIndex:(NSUInteger)index;
- (void)enumerateItems:(void(^)(id <WebItemProviderRegistrar> item, NSUInteger index))block;

@end

typedef void (^WebItemProviderFileLoadBlock)(NSArray<NSURL *> *);

WEBCORE_EXPORT @interface WebItemProviderPasteboard : NSObject<AbstractPasteboard>

+ (instancetype)sharedInstance;

@property (copy, nonatomic, nullable) NSArray<__kindof NSItemProvider *> *itemProviders;
@property (readonly, nonatomic) NSInteger numberOfItems;
@property (readonly, nonatomic) NSInteger changeCount;

// This will only be non-empty when an operation is being performed.
@property (readonly, nonatomic) NSArray<NSURL *> *allDroppedFileURLs;

@property (readonly, nonatomic) BOOL hasPendingOperation;
- (void)incrementPendingOperationCount;
- (void)decrementPendingOperationCount;

- (void)setItemProviders:(NSArray<__kindof NSItemProvider *> *)itemProviders dropSession:(nullable id<UIDropSession>)dropSession;

- (void)enumerateItemProvidersWithBlock:(void (^)(__kindof NSItemProvider *itemProvider, NSUInteger index, BOOL *stop))block;

// The given completion block is always dispatched on the main thread.
- (void)doAfterLoadingProvidedContentIntoFileURLs:(WebItemProviderFileLoadBlock)action;
- (void)doAfterLoadingProvidedContentIntoFileURLs:(WebItemProviderFileLoadBlock)action synchronousTimeout:(NSTimeInterval)synchronousTimeout;

@end

NS_ASSUME_NONNULL_END

#endif // TARGET_OS_IOS || (defined(TARGET_OS_VISION) && TARGET_OS_VISION)
