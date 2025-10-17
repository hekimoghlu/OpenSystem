/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#import <Foundation/NSString.h>

@interface NSString (NSStringAdditions)
+ (NSString *) mimeCharsetTagFromStringEncoding:(NSStringEncoding) encoding;

- (unsigned long) UTF8StringByteLength;

- (id) initWithBytes:(const void *) bytes encoding:(NSStringEncoding) encoding;
+ (id) stringWithBytes:(const void *) bytes encoding:(NSStringEncoding) encoding;

- (const char *) bytesUsingEncoding:(NSStringEncoding) encoding allowLossyConversion:(BOOL) lossy;
- (const char *) bytesUsingEncoding:(NSStringEncoding) encoding;

- (NSString *) stringByEncodingXMLSpecialCharactersAsEntities;
- (NSString *) stringByDecodingXMLSpecialCharacterEntities;

- (NSString *) stringByEscapingCharactersInSet:(NSCharacterSet *) set;

- (NSString *) stringByEncodingIllegalURLCharacters;
- (NSString *) stringByDecodingIllegalURLCharacters;
@end

@interface NSMutableString (NSMutableStringAdditions)
- (void) encodeXMLSpecialCharactersAsEntities;
- (void) decodeXMLSpecialCharacterEntities;

- (void) escapeCharactersInSet:(NSCharacterSet *) set;

- (void) encodeIllegalURLCharacters;
- (void) decodeIllegalURLCharacters;
@end
