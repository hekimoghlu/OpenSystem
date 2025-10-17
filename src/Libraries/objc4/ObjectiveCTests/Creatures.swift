/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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

// These are used to test objc_enumerateClasses()

import Foundation

@objc
enum CreatureSize: Int {
    case Unknown
    case Minascule
    case Small
    case Medium
    case Big
    case Huge
}

@objc
enum StripeColor: Int {
    case BlackAndOrange
    case GrayAndBlack
    case Plaid
}

@objc
protocol Creature {
    var name: String { get }
    var size: CreatureSize { get }
}

@objc
protocol Claws {
    func retract()
    func extend()
}

@objc
protocol Stripes {
    var stripeColor: StripeColor { get }
}

@objc(Animal)
class Animal: NSObject, Creature {
    var name: String { return "animal"; }
    var size: CreatureSize { return .Unknown }
}

@objc(Dog)
class Dog: Animal {
    override var name: String { return "dog"; }
}

@objc(Datschund)
class Datschund: Dog {
    override var name: String { return "datschund" }
    override var size: CreatureSize { return .Medium }
}

@objc(Terrier)
class Terrier: Dog {
    override var name: String { return "terrier" }
    override var size: CreatureSize { return .Small }
}

@objc(Labrador)
class Labrador: Dog {
    override var name: String { return "labrador" }
    override var size: CreatureSize { return .Medium }
}

@objc(Mastiff)
class Mastiff: Dog {
    override var name: String { return "mastiff" }
    override var size: CreatureSize { return .Big }
}

@objc(Cat)
class Cat: Animal {
    override var name: String { return "cat" }
}

@objc(Tabby)
class Tabby: Cat, Stripes {
    override var name: String { return "tabby" }
    override var size: CreatureSize { return .Small }
    var stripeColor: StripeColor { return .GrayAndBlack }
}

@objc(Lion)
class Lion: Cat {
    override var name: String { return "lion" }
    override var size: CreatureSize { return .Big }
}

@objc(Tiger)
class Tiger: Cat, Stripes {
    override var name: String { return "tiger" }
    override var size: CreatureSize { return .Big }
    var stripeColor: StripeColor { return .BlackAndOrange }
}

@objc(Elephant)
class Elephant: Animal {
    override var name: String { return "elephant" }
    override var size: CreatureSize { return .Huge }
}

@objc(Woozle)
class Woozle: Elephant, Stripes {
    override var name: String { return "woozle" }
    var stripeColor: StripeColor { return .Plaid }
}

extension Cat: Claws {
    func retract() {}
    func extend() {}
}
