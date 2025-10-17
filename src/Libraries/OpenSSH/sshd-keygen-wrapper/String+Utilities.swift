/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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

extension String {
  static let onlySafeShellCharacters = #/[-@%_+:,./A-Za-z0-9]*/#

  /// Return a quoted version of the string for copy-and-paste
  /// to a shell.
  var shellQuoted: String {
    if self == "" {
      return "''"
    }
    if let _ = try? Self.onlySafeShellCharacters.wholeMatch(in: self) {
      return self
    }
    var quoted = "'"
    for ch in self {
      switch ch {
      case "'":
        quoted.append("'\\''")
      default:
        quoted.append(ch)
      }
    }
    quoted.append("'")
    return quoted
  }

  /// Create a new string with the value initialized from
  /// an environmental variable.
  init?(fromEnvironmentVariable name: String) {
    guard let envp = getenv(name) else {
      return nil
    }
    self.init(cString: envp)
  }
}
