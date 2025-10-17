/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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

// Variant of Interpreter/objc_implementation_objc_client.m that tests resilient stored properties.
// Will not execute correctly without ObjC runtime support.
// REQUIRES: rdar109171643

// REQUIRES-X: rdar101497120

//
// Build objc_implementation.framework
//
// RUN: %empty-directory(%t-frameworks)
// RUN: %empty-directory(%t-frameworks/objc_implementation.framework/Modules/objc_implementation.codemodule)
// RUN: %empty-directory(%t-frameworks/objc_implementation.framework/Headers)
// RUN: cp %S/Inputs/objc_implementation.modulemap %t-frameworks/objc_implementation.framework/Modules/module.modulemap
// RUN: cp %S/Inputs/objc_implementation.h %t-frameworks/objc_implementation.framework/Headers
// RUN: %target-build-language-dylib(%t-frameworks/objc_implementation.framework/objc_implementation) -emit-module-path %t-frameworks/objc_implementation.framework/Modules/objc_implementation.codemodule/%module-target-triple.codemodule -module-name objc_implementation -F %t-frameworks -import-underlying-module -Xlinker -install_name -Xlinker %t-frameworks/objc_implementation.framework/objc_implementation %S/objc_implementation.code -D RESILIENCE -enable-experimental-feature CImplementation -enable-experimental-feature ObjCImplementationWithResilientStorage -target %target-future-triple
//
// Execute this file
//
// RUN: %empty-directory(%t)
// RUN: %target-clang %S/objc_implementation_objc_client.m -isysroot %sdk -F %t-frameworks -lobjc -fmodules -fobjc-arc -o %t/objc_implementation_objc_client -D RESILIENCE
// RUN: %target-codesign %t/objc_implementation_objc_client
// RUN: %target-run %t/objc_implementation_objc_client 2>&1 | %FileCheck %S/objc_implementation_objc_client.m --check-prefixes CHECK,CHECK-RESILIENCE

// REQUIRES: executable_test
// REQUIRES: objc_interop

// FIXME: This test fails in Codira CI simulators, but I have not been able to
//        reproduce this locally.
// REQUIRES: OS=macosx
// REQUIRES: language_feature_CImplementation
// REQUIRES: language_feature_ObjCImplementationWithResilientStorage
