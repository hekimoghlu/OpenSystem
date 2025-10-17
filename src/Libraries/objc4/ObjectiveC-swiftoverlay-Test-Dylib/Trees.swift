/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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
//  Trees.swift
//  Trees
//
//  Created by Alastair Houghton on 02/09/2021.
//  Copyright Â© 2021 Apple. All rights reserved.
//

import Foundation

@objc
protocol Tree {
    func name() -> String
    func isEvergreen() -> Bool
}

@objc
class Oak: NSObject, Tree {
    public func name() -> String { return "oak" }
    public func isEvergreen() -> Bool { return false }
}

@objc
class Birch: NSObject, Tree {
    public func name() -> String { return "birch" }
    public func isEvergreen() -> Bool { return false }
}

@objc
class Pine: NSObject, Tree {
    public func name() -> String { return "birch" }
    public func isEvergreen() -> Bool { return false }
}

@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
@_cdecl("testEnumerateClassesFromDylib")
public func testEnumerateClassesFromDylib() -> Bool {
    // This should enumerate the classes *in the dylib*
    let trees = objc_enumerateClasses().map{ "\($0)" }
    if trees == [ "Oak", "Birch", "Pine" ] {
        return true
    } else {
        print("FAILED: trees was \(trees)!")
        return false
    }
}
