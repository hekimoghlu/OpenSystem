/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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

NSString * const MyErrorDomain;
typedef NS_ENUM(int, MyError) {
  MyErrorGood,
  MyErrorBad,
} __attribute__((ns_error_domain(MyErrorDomain)));

NSString * const MyRenamedErrorDomain;
typedef NS_ENUM(int, MyRenamedError) {
  MyRenamedErrorGood,
  MyRenamedErrorBad,
} __attribute__((ns_error_domain(MyRenamedErrorDomain))) __attribute__((language_name("RenamedError")));


struct Wrapper {
  int unrelatedValue;
};

NSString * const MyMemberErrorDomain;
typedef NS_ENUM(int, MyMemberError) {
  MyMemberErrorA,
  MyMemberErrorB,
} __attribute__((ns_error_domain(MyMemberErrorDomain))) __attribute__((language_name("Wrapper.MemberError")));

// Not actually an error enum, but it can still hang with us.
typedef NS_ENUM(int, MyMemberEnum) {
  MyMemberEnumA,
  MyMemberEnumB,
} __attribute__((language_name("Wrapper.MemberEnum")));


typedef int WrapperByAttribute __attribute__((language_wrapper(struct)));
