/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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

// RUN: %empty-directory(%t)
// RUN: %target-language-frontend(mock-sdk: %clang-importer-sdk) %S/../Inputs/empty.code -typecheck -verify -emit-objc-header-path %t/empty.h
// RUN: %clang -F %clang-importer-sdk-path/frameworks -E -fobjc-arc -fmodules -isysroot %clang-importer-sdk-path -I %t %s | %FileCheck %s

// REQUIRES: objc_interop

#import "empty.h"

@class ABC; // CHECK-LABEL: @class ABC;
LANGUAGE_CLASS(abc) // CHECK-NEXT: __attribute__((objc_runtime_name(abc)))
@interface ABC // CHECK-NEXT: @interface
@end

@class DEF; // CHECK-LABEL: @class DEF;
LANGUAGE_CLASS_NAMED(def) // CHECK-NEXT: __attribute__((language_name(def)))
@interface DEF // CHECK-NEXT: @interface
@end

@protocol AAA; // CHECK-LABEL: @protocol AAA;
LANGUAGE_PROTOCOL(aaa) // CHECK-NEXT: __attribute__((objc_runtime_name(aaa)))
@protocol AAA // CHECK-NEXT: @protocol
@end

@protocol BBB; // CHECK-LABEL: @protocol BBB;
LANGUAGE_PROTOCOL_NAMED(bbb) // CHECK-NEXT: __attribute__((language_name(bbb)))
@protocol BBB // CHECK-NEXT: @protocol
@end
