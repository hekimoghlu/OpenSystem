/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

import CoreAnalytics
import Foundation
import System

struct SSHDWrapper {
  // MARK: - Strategy types
  typealias PlanTransformer = (Subprocess.Plan) -> Subprocess.Plan

  // MARK: - Dependencies
  let hostKeyPlanTransformer: PlanTransformer?
  let sshdPlanTransformer: PlanTransformer?
  let systemProperties: SystemPropertiesStrategy

  // MARK: - Instance methods
  func usage() {
    _ = try? FileDescriptor.standardError.writeAll(
      "Usage: sshd-keygen-wrapper\n".utf8)
  }

  /// sshd-keygen-wrapperâ€™s main entry point. First generates host
  /// keys, if not already present. Then builds `sshd` command
  /// line arguments appropriate for the platform and system
  /// configuration, and finally spawns `sshd`.
  func run(_ arguments: [String] = CommandLine.arguments) async throws {
    dump(arguments, name: "arguments")
    guard arguments.count <= 1 else {
      throw Error.unexpectedArgument(arguments[1])
    }

    let sshdArguments = ["-i"]
    let sshKeygen = systemProperties.pathPrefix.appending("bin/ssh-keygen")
    let generator = HostKeyGenerator(
      keygen: sshKeygen,
      directory: systemProperties.sshDirectory,
      plan: hostKeyPlanTransformer)
    for algorithm in HostKeyGenerator.Algorithm.allCases {
      do {
        if try await generator.generate(algorithm: algorithm) {
          logger.info("Generated \(algorithm.rawValue) host key")
        }
      } catch {
        logger.error("Failed to generate \(algorithm.rawValue) host key: \(error)")
      }
    }

    let sshd = systemProperties.pathPrefix.appending("sbin/sshd")
    var plan = Subprocess.Plan(path: sshd, arguments: sshdArguments)
    plan.flags = [.setExec]
    if let sshdPlanTransformer {
      plan = sshdPlanTransformer(plan)
    }
    let process = Subprocess(plan)
    let result = try await process.run()
    // only reachable during testing
    if !result.success {
      let errorString = try? process.errorString
      throw Error.sshdFailed(process.command, errorString ?? "unknown error", result)
    }
  }

  // MARK: - Initialization

  /// Creates object encapsulating the main logic for launching `sshd`.
  /// The parameters permit specifying optional dependencies for
  /// testing.
  /// - Parameters:
  ///   - hostKeyPlanTransformer: This function is given the
  ///     `Subprocess.Plan` for invocations of `ssh-keygen`, and
  ///     returns a modified plan that will be used.
  ///   - sshdPlanTransformer: As previous, but for invocations
  ///     of `sshd`.
  ///   - systemProperties: Provides properties representing
  ///     the systemâ€™s run time environment and configuration.
  init(
    hostKeyPlanTransformer: PlanTransformer? = nil,
    sshdPlanTransformer: PlanTransformer? = nil,
    systemProperties: SystemPropertiesStrategy = SystemProperties()
  ) {
    self.hostKeyPlanTransformer = hostKeyPlanTransformer
    self.sshdPlanTransformer = sshdPlanTransformer
    self.systemProperties = systemProperties
  }

  // MARK: - Errors
  enum Error: Swift.Error, CustomStringConvertible {
    case sshdFailed(String, String, Subprocess.Result)
    case unexpectedArgument(String)

    var description: String {
      return
        switch self
      {
      case .sshdFailed(let command, let error, let result):
        "\(command): \(result): \(error)"
      case .unexpectedArgument(let arg):
        "Unexpected argument: â€œ\(arg)â€."
      }
    }
  }
}
