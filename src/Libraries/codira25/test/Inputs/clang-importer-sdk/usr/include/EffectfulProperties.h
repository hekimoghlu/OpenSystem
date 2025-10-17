/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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

#define MAIN_ACTOR __attribute__((__language_attr__("@MainActor")))
#define ASYNC_NAME(X) __attribute__((language_async_name(X)));

#pragma clang assume_nonnull begin

@interface EffProps : NSObject

/////
// decls imported as an effectful property

-(void)getDogWithCompletion:(void (^)(NSObject*))completionHandler
  ASYNC_NAME("getter:doggo()");

-(void)obtainCat:(void (^)(NSObject* _Nullable_result, NSError* _Nullable))completionHandler
  ASYNC_NAME("getter:catto()");

-(void)checkAvailabilityWithCompletionHandler:(void (^)(BOOL isAvailable))completionHandler
  ASYNC_NAME("getter:available()");

// FIXME: this does not raise an error, and generates two identical properties!
// -(void)anotherExampleWithCompletionBlock:(void (^)(NSString *))block
//   ASYNC_NAME("getter:manifestedString()");
// -(void)finalExampleWithReplyTo:(void (^)(NSString *))block
//   ASYNC_NAME("getter:manifestedString()");

-(void)returnNothingWithCompletion:(void (^)(void))completionHandler
  ASYNC_NAME("getter:touch()");

-(void)nullableHandler:(void (^ _Nullable)(NSString *))completion
ASYNC_NAME("getter:fromNullableHandler()");

-(void)getMainDog:(MAIN_ACTOR void (^)(NSString *))completion
ASYNC_NAME("getter:mainDogProp()");
-(void)regularMainDog:(MAIN_ACTOR void (^)(NSString *))completion;

@end

@interface NotEffProps : NSObject

/////
// decls that are _not_ imported as an effectful property

// FIXME: even prior to objc imported effectful properties, this triggers an assert.
// -(NSObject*)getGreenDog __attribute__((language_name("getter:greenDoggie()")));

-(void)doSomethingSlow:(NSString *)operation completionHandler:(void (^)(NSInteger))handler
  ASYNC_NAME("getter:slow(self:)");

-(void)doSomethingDangerous:(NSString *)operation completionHandler:(void (^ _Nullable)(NSString *_Nullable, NSError * _Nullable))handler 
  ASYNC_NAME("getter:slowAndDanger(self:)");

// This means it can throw in two ways: produce an error synchronously, returning
// false and put error in out parameter. Or return true and error/success goes to completion handler.
// NOTE: no need to support this, but make sure we have a test case.
-(BOOL)getChicken:(void (^)(NSObject* _Nullable_result, NSError* _Nullable))completionHandler
       error: (NSError**)error
  ASYNC_NAME("getter:chicken()");


// plain throws, with `language_name`. should be ignored and imported like before.
-(NSObject* _Nullable)getCow1:(NSError**)error
  __attribute__((language_name("getter:noCow1()")));

// plain throws, with `language_async_name`. Sensible to support in the future, but should be ignored.
-(NSObject* _Nullable)getCow2:(NSError**)error ASYNC_NAME("getter:noCow2()");

@end

// make sure import-as-member does not also get turned into an effectful prop

struct __attribute__((language_name("Struct1"))) IAMStruct1 {};

extern double EffPropGetDogWithCompletion(const struct Struct1 *s, void (^myBlock)(NSObject*))
    ASYNC_NAME("getter:Struct1.dog(self:)");

#pragma clang assume_nonnull end
