/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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

// TEST_CONFIG ARCH=arm64e

import Darwin

var didFail = false

func fail(_ msg: String) {
    print("BAD: \(msg)")
    didFail = true
}

class Generic<T> {}

// Ensure that a Swift generic metaclass has properly signed isa/superclass
// pointers. Work with raw pointers to avoid retain/release on the class
// objects, which Swift ARC likes to do.
let genericRaw = unsafeBitCast(Generic<Int>.self, to: UnsafeRawPointer.self)

let RTLD_DEFAULT = UnsafeMutableRawPointer(bitPattern: -2)
typealias classToClassFn = @convention(c) (UnsafeRawPointer?) -> UnsafeRawPointer?

let object_getClassRaw = dlsym(RTLD_DEFAULT, "object_getClass")
let object_getClass = unsafeBitCast(object_getClassRaw, to: classToClassFn.self)

let class_getSuperClassRaw = dlsym(RTLD_DEFAULT, "class_getSuperclass")
let class_getSuperClass = unsafeBitCast(class_getSuperClassRaw, to: classToClassFn.self)

// Check for nil, but we're really checking for ptrauth failures in the call.

if object_getClass(object_getClass(genericRaw)) == nil {
    fail("metaclass of metaclass is nil")
}

if class_getSuperClass(object_getClass(genericRaw)) == nil {
    fail("superclass of metaclass is nil")
}

if !didFail {
    print("OK:", #file.split(separator: "/").last!)
}
