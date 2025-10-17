/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

@import Foundation;

#ifndef NS_LANGUAGE_NAME(Name)
#  define NS_LANGUAGE_NAME(Name) __attribute__((language_name(#Name)))
#endif

@interface SEGreebieArray : NSObject
@end

typedef NS_OPTIONS(NSUInteger, OMWWobbleOptions) {
  OMWWobbleSideToSide = 0x01,
  OMWWobbleBackAndForth = 0x02,
  OMWWobbleToXMLHex = 0x04
};

@interface OmitNeedlessWords : NSObject
-(void)jumpToUrl:(nonnull NSURL *)url;
-(void)jumpToGuid:(nonnull NSGUID *)guid;
-(void)jumpAgainToGUID:(nonnull NSGUID *)guid;
-(BOOL)objectIsCompatibleWithObject:(nonnull id)other;
-(void)insetByX:(NSInteger)x y:(NSInteger)y;
-(void)setIndirectlyToValue:(nonnull id)object;
-(void)jumpToTop:(nonnull id)sender;
-(void)removeWithNoRemorse:(nonnull id)object;
-(void)bookmarkWithURLs:(nonnull NSArray<NSURL *> *)urls;
-(void)bookmarkWithGUIDs:(nonnull NSArray<NSGUID *> *)guids;
-(void)saveToURL:(nonnull NSURL *)url forSaveOperation:(NSInteger)operation;
-(void)saveToGUID:(nonnull NSGUID *)guid forSaveOperation:(NSInteger)operation;
-(void)indexWithItemNamed:(nonnull NSString *)name;
-(void)methodAndReturnError:(NSError **)error;
-(nullable Class)typeOfString:(nonnull NSString *)string;
-(nullable Class)typeOfNamedString:(nonnull NSString *)string;
-(nullable Class)typeOfTypeNamed:(nonnull NSString *)string;
-(void)appendWithContentsOfString:(nonnull NSString *)string;
-(nonnull id)objectAtIndexedSubscript:(NSUInteger)idx;
-(void)exportPresetsBestMatchingString:(nonnull NSString *)string;
-(void)isCompatibleWithString:(nonnull NSString *)string;
-(void)addObjectValue:(nonnull id)object;
-(nonnull OmitNeedlessWords *)wordsBySlobbering:(nonnull NSString *)string;
-(void)drawPolygonWithPoints:(const NSPoint[])points count:(NSInteger)count;
-(void)drawFilledPolygonWithPoints:(NSPointArray)points count:(NSInteger)count;
-(void)drawGreebies:(nonnull SEGreebieArray*)greebies;
-(void)doSomethingBoundBy:(NSInteger)value;
-(void)doSomethingSeparatedBy:(NSInteger)value;
+(nonnull OmitNeedlessWords *)currentOmitNeedlessWords;
+(void)setCurrentOmitNeedlessWords:(nonnull OmitNeedlessWords *)value;
-(void)compilerPlugInValue:(NSInteger)value;
-(void)wobbleWithOptions:(OMWWobbleOptions)options;
@end

@interface ABCDoodle : NSObject
@property (nonatomic,copy,nonnull) NSArray<ABCDoodle *> *doodles;
-(void)addDoodle:(nonnull ABCDoodle *)doodle;
@end

@protocol OMWLanding
-(void)flipLanding;
@end

@protocol OMWWiggle
-(void)joinSub;
@property (readonly) NSInteger conflictingProp1 NS_LANGUAGE_NAME(wiggleProp1);
@end

@protocol OMWWaggle
@property (readonly) NSInteger conflictingProp1 NS_LANGUAGE_NAME(waggleProp1);
@end

@interface OMWSuper : NSObject <OMWWiggle>
-(void)jumpSuper;
@property (readonly) NSInteger conflictingProp1;
@end

@interface OMWSub : OMWSuper <OMWWaggle>
-(void)jumpSuper;
-(void)joinSub;
@property (readonly) NSInteger conflictingProp1;
@end

@interface OMWObjectType : NSObject
-(void)_enumerateObjectTypesWithHandler:(nonnull void (^)(void))handler;
@end

@interface OMWTerrifyingGarbage4DTypeRefMask_t : NSObject
-(void)throwGarbageAway;
-(void)throwGarbage4DAwayHarder;
-(void)throwGarbage4DTypeRefMask_tAwayHardest;
-(void)burnGarbage;
-(void)carefullyBurnGarbage4D;
-(void)veryCarefullyBurnGarbage4DTypeRefMask_t;
@end
