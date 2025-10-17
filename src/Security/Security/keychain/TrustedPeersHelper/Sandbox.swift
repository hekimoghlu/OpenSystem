/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
import Foundation
import os.log
import SandboxPrivate

private let logger = Logger(subsystem: "com.apple.security.trustedpeers", category: "container")

// MARK: Sandbox

struct Sandbox {
    #if os(macOS)
    static var sandboxParameters: [String: String] {
        let homeDirectory: String

        if let home = NSHomeDirectory().realpath {
            homeDirectory = home
        } else {
            logger.info("User does not have a home directory! -- Falling back to /private/var/empty")
            homeDirectory = "/private/var/empty"
        }

        guard let tempDirectory = Self._confstr(_CS_DARWIN_USER_TEMP_DIR)?.realpath else {
            fatalError("Unable to read _CS_DARWIN_USER_TEMP_DIR!")
        }

        guard let cacheDirectory = Self._confstr(_CS_DARWIN_USER_CACHE_DIR)?.realpath else {
            fatalError("Unable to read _CS_DARWIN_USER_CACHE_DIR!")
        }

        return [
            "_DARWIN_USER_CACHE": cacheDirectory,
            "_DARWIN_USER_TEMP": tempDirectory,
            "_HOME": homeDirectory,
        ]
    }

    static func enterSandbox(identifier: String, profile: String) {
        guard _set_user_dir_suffix(identifier) else {
            fatalError("_set_user_dir_suffix() failed!")
        }

        _sandboxInit(profile: profile, parameters: sandboxParameters)
    }
    #else
    static func enterSandbox(identifier: String) {
        guard _set_user_dir_suffix(identifier) else {
            fatalError("_set_user_dir_suffix() failed!")
        }

        guard (Self._confstr(_CS_DARWIN_USER_TEMP_DIR)) != nil else {
            fatalError("Unable to read _CS_DARWIN_USER_TEMP_DIR!")
        }
    }
    #endif

    #if os(macOS)
    private static func _flatten(_ dictionary: [String: String]) -> [String] {
        var result = [String]()

        dictionary.keys.forEach { key in
            guard let value = dictionary[key] else {
                return
            }

            result.append(key)
            result.append(value)
        }

        return result
    }

    private static func _sandboxInit(profile: String, parameters: [String: String]) {
        var sbError: UnsafeMutablePointer<Int8>?
        let flatParameters = _flatten(parameters)
        logger.debug("Sandbox parameters: \(String(describing: parameters))")

        withArrayOfCStrings(flatParameters) { ptr -> Void in
            let result = sandbox_init_with_parameters(profile, UInt64(SANDBOX_NAMED), ptr, &sbError)
            guard result == 0 else {
                guard let sbError = sbError else {
                    fatalError("sandbox_init_with_parameters failed! (no error)")
                }

                fatalError("sandbox_init_with_parameters failed!: [\(String(cString: sbError))]")
            }
        }

        _ = sbError
    }
    #endif

    private static func _confstr(_ name: Int32) -> String? {
        var directory = Data(repeating: 0, count: Int(PATH_MAX))

        return directory.withUnsafeMutableBytes { body -> String? in
            guard let ptr = body.bindMemory(to: Int8.self).baseAddress else {
                logger.error("failed to bind memory")
                return nil
            }
            errno = 0
            let status = confstr(name, ptr, Int(PATH_MAX))

            guard status > 0 else {
                logger.error("confstr \(name) failed: \(errno)")
                return nil
            }
            return String(cString: ptr)
        }
    }
}

// For calling C functions with arguments like: `const char *const parameters[]`
private func withArrayOfCStrings<R>(_ args: [String], _ body: ([UnsafePointer<CChar>?]) -> R) -> R {
    let mutableStrings = args.map { strdup($0) }
    var cStrings = mutableStrings.map { UnsafePointer($0) }

    defer { mutableStrings.forEach { free($0) } }

    cStrings.append(nil)

    return body(cStrings)
}

private extension String {
    var realpath: String? {
        let retValue: String?

        guard let real = Darwin.realpath(self, nil) else {
            return nil
        }

        retValue = String(cString: real)

        real.deallocate()

        return retValue
    }
}
