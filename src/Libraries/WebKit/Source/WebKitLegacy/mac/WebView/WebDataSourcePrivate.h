/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#import <WebKitLegacy/WebDataSource.h>

@protocol WebDataSourcePrivateDelegate
@required
- (void)dataSourceMemoryMapped;
- (void)dataSourceMemoryMapFailed;
@end

@interface WebDataSource (WebPrivate)

#if !TARGET_OS_IPHONE
- (NSFileWrapper *)_fileWrapperForURL:(NSURL *)URL;
#endif
- (void)_addSubframeArchives:(NSArray *) archives;
- (NSError *)_mainDocumentError;
- (NSString *)_responseMIMEType;

- (void)_setDeferMainResourceDataLoad:(BOOL)flag;

#if TARGET_OS_IPHONE
- (void)_setOverrideTextEncodingName:(NSString *)encoding;
#endif
- (void)_setAllowToBeMemoryMapped;
- (void)setDataSourceDelegate:(NSObject<WebDataSourcePrivateDelegate> *)dataSourceDelegate;
- (NSObject<WebDataSourcePrivateDelegate> *)dataSourceDelegate;

#if TARGET_OS_IPHONE
@property (nonatomic, readonly) NSDictionary *_quickLookContent;
#endif

@end
