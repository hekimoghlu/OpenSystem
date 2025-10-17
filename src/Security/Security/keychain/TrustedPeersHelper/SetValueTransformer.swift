/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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

import CoreData
import Foundation

private let logger = Logger(subsystem: "com.apple.security.trustedpeers", category: "SetValueTransformer")

@objc(SetValueTransformer)
class SetValueTransformer: ValueTransformer {

    override class func transformedValueClass() -> AnyClass {
        return NSData.self
    }

    override class func allowsReverseTransformation() -> Bool {
        return true
    }

    override func transformedValue(_ value: Any?) -> Any? {
        do {
            guard let value = value else {
                return nil
            }
            return try NSKeyedArchiver.archivedData(withRootObject: value, requiringSecureCoding: true)
        } catch {
            logger.info("Failed to serialize a Set: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    override func reverseTransformedValue(_ value: Any?) -> Any? {
        do {
            guard let dataOp = value as? Data? else {
                return nil
            }
            guard let data = dataOp else {
                return nil
            }

            let unarchiver = try NSKeyedUnarchiver(forReadingFrom: data)
            return unarchiver.decodeObject(of: [NSSet.self, NSString.self], forKey: NSKeyedArchiveRootObjectKey)
        } catch {
            logger.info("Failed to deserialize a purported Set: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    static let name = NSValueTransformerName(rawValue: "SetValueTransformer")
}
