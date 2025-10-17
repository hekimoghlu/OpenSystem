/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

//
//  SecNSAdditions.h
//  Security
//

#ifndef _SECNSADDITIONS_H_
#define _SECNSADDITIONS_H_

#import <Foundation/Foundation.h>

static inline BOOL NSIsEqualSafe(NSObject* obj1, NSObject* obj2) {
    return obj1 == nil ? (obj2 == nil) : [obj1 isEqual:obj2];
}


// MARK: NSArray

@interface NSArray (compactDescription)
- (NSMutableString*) concatenateWithSeparator: (NSString*) separator;
@end

@interface NSDictionary (SOSDictionaryFormat)
- (NSString*) compactDescription;
@end

@interface NSMutableDictionary (FindAndRemove)
-(NSObject*)extractObjectForKey:(NSString*)key;
@end

// MARK: NSSet

@interface NSSet (Emptiness)
- (bool) isEmpty;
@end

@interface NSSet (HasElements)
- (bool) containsElementsNotIn: (NSSet*) other;
@end

@interface NSSet (compactDescription)
- (NSString*) shortDescription;
@end

@interface NSSet (Stringizing)
- (NSString*) sortedElementsJoinedByString: (NSString*) separator;
- (NSString*) sortedElementsTruncated: (NSUInteger) length JoinedByString: (NSString*) separator;
@end



// MARK: NSString

static inline NSString* asNSString(NSObject* object) {
    return [object isKindOfClass:[NSString class]] ? (NSString*) object : nil;
}

@interface NSString (FileOutput)
- (void) writeTo: (FILE*) file;
- (void) writeToStdOut;
- (void) writeToStdErr;
@end

// MARK: NSData

static inline NSData* asNSData(NSObject* object) {
    return [object isKindOfClass:[NSData class]] ? (NSData*) object : nil;
}

@interface NSData (Hexinization)
- (NSString*) asHexString;
@end

@interface NSMutableData (filledAndClipped)
+ (instancetype) dataWithSpace: (NSUInteger) initialSize DEREncode: (uint8_t*(^)(size_t size, uint8_t *buffer)) initialization;
@end

#endif /* SecNSAdditions_h */
