/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#import <WebKitLegacy/DOMNode.h>

@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMCharacterData : DOMNode
@property (copy) NSString *data;
@property (readonly) unsigned length;

- (NSString *)substringData:(unsigned)offset length:(unsigned)length WEBKIT_AVAILABLE_MAC(10_5);
- (void)appendData:(NSString *)data;
- (void)insertData:(unsigned)offset data:(NSString *)data WEBKIT_AVAILABLE_MAC(10_5);
- (void)deleteData:(unsigned)offset length:(unsigned)length WEBKIT_AVAILABLE_MAC(10_5);
- (void)replaceData:(unsigned)offset length:(unsigned)length data:(NSString *)data WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMCharacterData (DOMCharacterDataDeprecated)
- (NSString *)substringData:(unsigned)offset :(unsigned)length WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)insertData:(unsigned)offset :(NSString *)data WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)deleteData:(unsigned)offset :(unsigned)length WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)replaceData:(unsigned)offset :(unsigned)length :(NSString *)data WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
