/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

//
//  trustdReset.swift
//  Security
//
//

import ArgumentParserInternal
import Foundation
import OSLog

public struct trustdReset: ParsableCommand {
    public static var configuration = CommandConfiguration(
        commandName: "reset",
        abstract: "Delete files to reset trustd state",
        subcommands: [
            resetPublic.self,
            resetPrivate.self,
        ]
    )

    public init() { }
}

public struct resetPublic: ParsableCommand {
    public static var configuration = CommandConfiguration(
        commandName: "public",
        abstract: "Reset public trustd files")

    public func run() throws {
        guard os_variant_allows_internal_security_policies("com.apple.security") else {
            print("Cannot reset. Not an internal build.")
            Foundation.exit(1)
        }
        Logger().notice("Deleting /var/protected/trustd/")
        /* change permissions so we can delete the directory */
        if chmod("/private/var/protected/trustd", S_IRWXU | S_IRWXG | S_IRWXO) != 0 {
            let errStr = strerror(errno)!
            print("Failed to change directory permissions: \(errStr)")
            Foundation.exit(1)
        }
        defer {
            /* change permissions back */
            if chmod("/private/var/protected/trustd", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) != 0 {
                let errStr = strerror(errno)!
                print("Failed to change directory permissions back: \(errStr)")
                Foundation.exit(1)
            }
        }
        /* delete the directory contents */
        do {
            try FileManager().removeItem(atPath: "/private/var/protected/trustd/SupplementalsAssets")
            try FileManager().removeItem(atPath: "/private/var/protected/trustd/valid.sqlite3")
            try FileManager().removeItem(atPath: "/private/var/protected/trustd/valid.sqlite3-shm")
            try FileManager().removeItem(atPath: "/private/var/protected/trustd/valid.sqlite3-wal")
            try FileManager().removeItem(atPath: "/private/var/protected/trustd/pinningrules.sqlite3")
        } catch {
            print("Failed to delete directory: \(error)")
            Foundation.exit(1)
        }
    }
    public init() { }
}

public struct resetPrivate: ParsableCommand {
    public static var configuration = CommandConfiguration(
        commandName: "private",
        abstract: "Reset private trustd files")

    public func run() throws {
        guard os_variant_allows_internal_security_policies("com.apple.security") else {
            print("Cannot reset. Not an internal build.")
            Foundation.exit(1)
        }
        Logger().notice("Deleting /var/protected/trustd/private")
        /* change permissions so we can delete the directory */
        if chmod("/private/var/protected/trustd", S_IRWXU | S_IRWXG | S_IRWXO) != 0 {
            let errStr = strerror(errno)!
            print("Failed to change directory permissions: \(errStr)")
            Foundation.exit(1)
        }

        defer {
            /* change permissions back */
            if chmod("/private/var/protected/trustd", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) != 0 {
                let errStr = strerror(errno)!
                print("Failed to change directory permissions back: \(errStr)")
                Foundation.exit(1)
            }
        }

        /* delete the directory */
        do {
            try FileManager().removeItem(atPath: "/private/var/protected/trustd/private/")
        } catch {
            print("Failed to delete directory: \(error)")
            Foundation.exit(1)
        }
    }
    public init() { }
}
