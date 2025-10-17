/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
// Provide type safe address primitives

protocol Address: Comparable, Strideable, CustomStringConvertible {
    var value: UInt64 { get }
    init(_ : UInt64)
}

extension Address {
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.value < rhs.value
    }
    func distance(to other: Self) -> Int64 {
        return Int64(bitPattern:other.value &- value)
    }
    func advanced(by n: Int64) -> Self {
        return Self(value &+ UInt64(bitPattern:n))
    }
    var pageAligned: Bool { return (self.value & UInt64(vm_page_mask)) == 0 }

    var description: String {
        return "0x\(String(self.value, radix: 16, uppercase: true))"
    }
}

struct PreferredAddress: Address {
    init(_ value: Int64) {
        self.value = UInt64(bitPattern:value)
    }
    init(_ value: UInt64) {
        self.value = value
    }
    let value: UInt64
    static func +(left: PreferredAddress, right: UInt64) -> PreferredAddress {
        return PreferredAddress(left.value + right)
    }
    static func -(left: PreferredAddress, right: PreferredAddress) -> UInt64 {
        return left.value - right.value
    }
    static func +(left: PreferredAddress, right: Slide) -> RebasedAddress {
        // It is possible (but rare) for the preferred load addres to be above the rebased address, resulting in a negative slide, so allow overflow
        return RebasedAddress(left.value &+ right.value)
    }
}

struct RebasedAddress: Address {
    init(_ value: Int64) {
        self.value = UInt64(bitPattern:value)
    }
    init?(_ value: Int64?) {
        guard let value else { return nil }
        self.value = UInt64(bitPattern:value)
    }
    init(_ value: UInt64) {
        self.value = value
    }
    init(_ value: UInt32) {
        self.value = UInt64(value)
    }
    let value: UInt64
    
    static func -(left: RebasedAddress, right: PreferredAddress) -> Slide {
        // It is possible (but rare) for the preferred load addres to be above the rebased address, resulting in a negative slide, so allow overflow
        return Slide(left.value &- right.value)
    }
}

struct Slide {
    init(_ value:UInt64) {
        self.value = value
    }
    let value: UInt64
}
