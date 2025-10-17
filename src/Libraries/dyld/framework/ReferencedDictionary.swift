/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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


// This is a dictionary which is a class instead of struct. We do that so it can be used in contexts where we need to mutate the dictionary
// without mutating the struct containing. The main use case for this is in the remote process notification enums. In particular, we want to
// have the `.connected` state have an associated set of values which are the live notifiers, which can be mutated without modifying the state
// object itself as the only way for the dispatch source cancel handler is to capture it at creation time. Since it will be captured by value we
// need the by reference indirection for it to pickup notifier change registration.

// This pattern is not inherently concurrency safe, so it must be protected by some external mecanism such a serial dispatch queue.

class ReferencedDictionary<K : Hashable,V>: Collection  {
    subscript(position: Dictionary<K, V>.Index) -> Dictionary<K, V>.Element {
        //TODO: We should implement _read, but it ICEs the compiler and this is not perf sensitive
        get {
            return dictionary[position]
        }
    }
    subscript(key: K) -> V? {
        get {
            return dictionary[key]
        }
        set {
            dictionary[key] = newValue
        }
    }

    func index(after i: Dictionary<K, V>.Index) -> Dictionary<K, V>.Index {
        dictionary.index(after:i)
    }
    private var dictionary: Dictionary<K,V> = [:]
    typealias Element = Dictionary<K,V>.Element
    var startIndex:Dictionary<K,V>.Index {
        return dictionary.startIndex
    }
    var endIndex:Dictionary<K,V>.Index {
        return dictionary.endIndex
    }
    func removeAll() {
        dictionary.removeAll()
    }
    var isEmpty: Bool {
        return dictionary.isEmpty
    }
    @discardableResult
    func removeValue(forKey key: K) -> V? {
        dictionary.removeValue(forKey: key)
    }
}
