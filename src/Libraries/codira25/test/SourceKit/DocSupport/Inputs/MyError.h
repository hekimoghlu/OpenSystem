/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

#define NS_ERROR_ENUM(_type, _name, _domain)  \
  enum _name : _type _name; enum __attribute__((ns_error_domain(_domain))) _name : _type

@class NSString;
extern const NSString *const MyErrorDomain;
/// This is my cool error code.
typedef NS_ERROR_ENUM(int, MyErrorCode, MyErrorDomain) {
  /// This is first error.
  MyErrFirst,
  /// This is second error.
  MyErrSecond,
};
