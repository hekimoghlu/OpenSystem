/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
import System

// Swift's FilePath works completely in process without making syscalls, so it cannot resolve symlinks.
// This matches dyld's usage, where we need to resolve symlinks iin archives or the shared cache, not
// an actual filesystem, so we can extended it by passing in a set of symlinks and/or a base path to
// use resolving a path.
internal extension FilePath {
    mutating func resolving(from path: FilePath = "", symlinks:[FilePath:FilePath]? = nil) {
        if #available(macOS 12.0, *) {
            self = path.pushing(self)
            self.lexicallyNormalize()
            guard let symlinks else { return }
            for _ in 0..<MAXSYMLINKS {
                var foundLink = false
                for (source, target) in symlinks {
                    guard self.starts(with:source) else { continue }
                    self.removeLastComponent()
                    self.push(target)
                    self.lexicallyNormalize()
                    foundLink = true
                }
                guard foundLink else { break }
            }
        } else {
            fatalError("This is not supported for backdeployment, but needs to build with a reduced minOS in some environments")
        }
    }
    func resolved(from path: FilePath = "", symlinks:[FilePath:FilePath]? = nil) -> FilePath {
        var result = self
        result.resolving(from: path, symlinks: symlinks)
        return result
    }
}
