/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
// SSReadingList is API, not SPI, but we need to redeclare it. SafariServices depends on WebKit
// headers, so a WebKit dependency on SafariServices headers forms a dependency cycle.
@interface SSReadingList : NSObject
+ (SSReadingList *)defaultReadingList;
+ (BOOL)supportsURL:(NSURL *)URL;
- (BOOL)addReadingListItemWithURL:(NSURL *)URL title:(NSString *)title previewText:(NSString *)previewText error:(NSError **)error;
@end
