/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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

import Darwin
import System

extension FilePath {
  /// Check for the existance of a file at this path.
  /// - Returns: true if the path exists and it is a regular file,
  ///   else false.
  func exists() -> Bool {
    var st = stat()
    let rc = stat(self.string, &st)
    let e = Errno(rawValue: Darwin.errno)
    if rc == 0 && (st.st_mode & S_IFMT) == S_IFREG {
      return true
    } else if rc == 0 {
      let mode = String(st.st_mode, radix: 16, uppercase: false)
      logger.error("\(self): Not a regular file, mode=\(mode)")
    } else if e != .noSuchFileOrDirectory {
      logger.error("\(self): [\(e.rawValue): \(e)]")
    }
    return false
  }
}
