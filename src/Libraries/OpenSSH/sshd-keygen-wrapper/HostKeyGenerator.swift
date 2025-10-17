/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

struct HostKeyGenerator {
  // MARK: - Properties
  let keygen: FilePath
  let directory: FilePath
  let planTransformer: SSHDWrapper.PlanTransformer?

  // MARK: - Instance methods
  func generate(algorithm: Algorithm) async throws -> Bool {
    let path = directory.appending("ssh_host_\(algorithm)_key")
    if path.exists() {
      return false
    }
    var plan = Subprocess.Plan(
      path: keygen,
      arguments: ["-q", "-t", algorithm.rawValue, "-f", path.string, "-N", "", "-C", ""],
      inputDisposition: .null,
      outputDisposition: .bytes,
      errorDisposition: .bytes
    )
    if let planTransformer {
      plan = planTransformer(plan)
    }
    let process = Subprocess(plan)
    let result = try await process.run()
    if !result.success {
      let errorString = try? process.errorString
      throw Error.commandFailed(process.command, errorString ?? "unknown error", result)
    }
    return true
  }

  // MARK: - Initilization
  init(keygen: FilePath, directory: FilePath, plan: SSHDWrapper.PlanTransformer? = nil) {
    self.keygen = keygen
    self.directory = directory
    self.planTransformer = plan
  }

  // MARK: - Supporting types
  enum Algorithm: String, CaseIterable {
    case dsa, ecdsa, ed25519, rsa
  }

  enum Error: Swift.Error, CustomStringConvertible {
    case commandFailed(String, String, Subprocess.Result)

    var description: String {
      return
        switch self
      {
      case .commandFailed(let command, let error, let result):
        "\(command): \(result): \(error)"
      }
    }
  }
}

