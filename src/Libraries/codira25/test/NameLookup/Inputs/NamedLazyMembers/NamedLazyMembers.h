/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

@protocol Doer

- (void)doSomeWork;
- (void)doSomeWorkWithSpeed:(int)s;
- (void)doSomeWorkWithSpeed:(int)s thoroughness:(int)t
  NS_LANGUAGE_NAME(doVeryImportantWork(speed:thoroughness:));
- (void)doSomeWorkWithSpeed:(int)s alacrity:(int)a
  NS_LANGUAGE_NAME(doSomeWorkWithSpeed(speed:levelOfAlacrity:));

// These we are generally trying to not-import, via laziness.
- (void)goForWalk;
- (void)takeNap;
- (void)eatMeal;
- (void)tidyHome;
- (void)callFamily;
- (void)singSong;
- (void)readBook;
- (void)attendLecture;
- (void)writeLetter;

@end


// Don't conform to the protocol; that loads all protocol members.
@interface SimpleDoer

- (instancetype)initWithValue: (int)value;

// These are names we're hoping don't interfere with Doer, above.
+ (SimpleDoer*)Doer;
+ (SimpleDoer*)DoerOfNoWork;

- (void)simplyDoSomeWork;
- (void)simplyDoSomeWorkWithSpeed:(int)s;
- (void)simplyDoSomeWorkWithSpeed:(int)s thoroughness:(int)t
  NS_LANGUAGE_NAME(simplyDoVeryImportantWork(speed:thoroughness:));
- (void)simplyDoSomeWorkWithSpeed:(int)s alacrity:(int)a
  NS_LANGUAGE_NAME(simplyDoSomeWorkWithSpeed(speed:levelOfAlacrity:));

// Make sure that language_private correctly adds the '__' prefix.
- (void)count __attribute__((language_private));
- (void)objectForKey:(NSObject *)key __attribute__((language_private));

// These we are generally trying to not-import, via laziness.
- (void)simplyGoForWalk;
- (void)simplyTakeNap;
- (void)simplyEatMeal;
- (void)simplyTidyHome;
- (void)simplyCallFamily;
- (void)simplySingSong;
- (void)simplyReadBook;
- (void)simplyAttendLecture;
- (void)simplyWriteLetter;

@end


// Don't conform to the protocol; that loads all protocol members.
@interface SimpleDoer (Category)
- (void)categoricallyDoSomeWork;
- (void)categoricallyDoSomeWorkWithSpeed:(int)s;
- (void)categoricallyDoSomeWorkWithSpeed:(int)s thoroughness:(int)t
  NS_LANGUAGE_NAME(categoricallyDoVeryImportantWork(speed:thoroughness:));
- (void)categoricallyDoSomeWorkWithSpeed:(int)s alacrity:(int)a
  NS_LANGUAGE_NAME(categoricallyDoSomeWorkWithSpeed(speed:levelOfAlacrity:));

// These we are generally trying to not-import, via laziness.
- (void)categoricallyGoForWalk;
- (void)categoricallyTakeNap;
- (void)categoricallyEatMeal;
- (void)categoricallyTidyHome;
- (void)categoricallyCallFamily;
- (void)categoricallySingSong;
- (void)categoricallyReadBook;
- (void)categoricallyAttendLecture;
- (void)categoricallyWriteLetter;

@end


@protocol MirroredBase
+ (void)mirroredBaseClassMethod;
- (void)mirroredBaseInstanceMethod;
@end

@protocol MirroredDoer <MirroredBase>
+ (void)mirroredDerivedClassMethod;
- (void)mirroredDerivedInstanceMethod;
@end

@interface MirroringDoer : NSObject<MirroredDoer>
- (void)unobtrusivelyGoForWalk;
- (void)unobtrusivelyTakeNap;
- (void)unobtrusivelyEatMeal;
- (void)unobtrusivelyTidyHome;
- (void)unobtrusivelyCallFamily;
- (void)unobtrusivelySingSong;
- (void)unobtrusivelyReadBook;
- (void)unobtrusivelyAttendLecture;
- (void)unobtrusivelyWriteLetter;
@end

@interface DerivedFromMirroringDoer : MirroringDoer
@end

@interface SimilarlyNamedThings
- (void)doSomething:(double)x;
- (void)doSomething:(double)x celsius:(double)y;
- (void)doSomething:(double)x fahrenheit:(double)y using:(void (^)(void))block;
@end

@interface SimpleDoerSubclass : SimpleDoer
- (void)simplyDoSomeWorkWithSpeed:(int)s thoroughness:(int)t
  NS_LANGUAGE_NAME(simplyDoVeryImportantWork(speed:thoroughness:));

- (void)exuberantlyGoForWalk;
- (void)exuberantlyTakeNap;
- (void)exuberantlyEatMeal;
- (void)exuberantlyTidyHome;
- (void)exuberantlyCallFamily;
- (void)exuberantlySingSong;
- (void)exuberantlyReadBook;
- (void)exuberantlyAttendLecture;
- (void)exuberantlyWriteLetter;
@end

@protocol PrivateMethods <NSObject>
- (void)count;
- (void)objectForKey:(NSObject *)key;
@end

@interface PrivateDoer
- (void)count;
- (void)objectForKey:(NSObject *)key;
@end

@interface PrivateDoer(Category) <PrivateMethods>
@end

typedef NSString * const SimpleDoerMode NS_TYPED_ENUM NS_LANGUAGE_NAME(SimpleDoer.Mode);
typedef NSString * const SimpleDoerKind NS_TYPED_ENUM NS_LANGUAGE_NAME(SimpleDoer.Kind);
