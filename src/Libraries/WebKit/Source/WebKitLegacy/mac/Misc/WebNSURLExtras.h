/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#import <Foundation/Foundation.h>

// FIXME: Consider renaming to _web_ from _webkit_ once identically-named methods are no longer present in Foundation.

@interface NSURL (WebNSURLExtras)

// Deprecated, as it ignores URL parsing errors.
// Please use the _webkit_URLWithUserTypedString instead.
+ (NSURL *)_web_URLWithUserTypedString:(NSString *)string;

// Return value of nil means error in URL parsing.
+ (NSURL *)_webkit_URLWithUserTypedString:(NSString *)string;

+ (NSURL *)_web_URLWithDataAsString:(NSString *)string;
+ (NSURL *)_web_URLWithDataAsString:(NSString *)string relativeToURL:(NSURL *)baseURL;

- (NSData *)_web_originalData;
- (NSString *)_web_originalDataAsString;
- (const char*)_web_URLCString;

- (NSString *)_web_hostString;

- (NSString *)_web_userVisibleString;

- (BOOL)_web_isEmpty;

- (NSURL *)_webkit_canonicalize;
- (NSURL *)_webkit_canonicalize_with_wtf;
- (NSURL *)_webkit_URLByRemovingFragment;
- (NSURL *)_web_URLByRemovingUserInfo;

- (BOOL)_webkit_isJavaScriptURL;
- (BOOL)_webkit_isFileURL;
- (NSString *)_webkit_scriptIfJavaScriptURL;

- (NSString *)_webkit_suggestedFilenameWithMIMEType:(NSString *)MIMEType;

- (NSURL *)_webkit_URLFromURLOrSchemelessFileURL;

@end

@interface NSString (WebNSURLExtras)

- (BOOL)_web_isUserVisibleURL;

// Deprecated as it ignores URL parsing errors.
// Please use _webkit_decodeHostName instead.
// Turns funny-looking ASCII form into Unicode, returns self if no decoding needed.
- (NSString *)_web_decodeHostName;

// Deprecated as it ignores URL parsing errors.
// Please use the _webkit_encodeHostName instead.
// Turns Unicode into funny-looking ASCII form, returns self if no encoding needed.
- (NSString *)_web_encodeHostName;

// Return value of nil means error in URL parsing.
- (NSString *)_webkit_decodeHostName;
- (NSString *)_webkit_encodeHostName;

- (BOOL)_webkit_isJavaScriptURL;
- (BOOL)_webkit_isFileURL;
- (BOOL)_webkit_looksLikeAbsoluteURL;
- (NSRange)_webkit_rangeOfURLScheme;
- (NSString *)_webkit_scriptIfJavaScriptURL;

@end
