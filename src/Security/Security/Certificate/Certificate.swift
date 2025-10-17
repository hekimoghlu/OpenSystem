/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
//  Certificate.swift
//  Security
//
//

import Foundation

struct Certificate {
    var secCertificate: UnsafeMutablePointer<SecCertificate>
    var privateKey: Data

    init(cert: Data) { }
    init(cert: Data, privateKey: Data) { }

    /// Return the DER representation of an X.509 certificate.
    ///
    ///
    func copyData() -> Data { }

    /// Return a string representing summary.
    ///
    ///
    func subjectSummary() -> String { }

    /// Returns the common name of the subject of a given certificate.
    ///
    ///
    func commonName() throws -> [String] { }

    /// Returns an array of zero or more email addresses for the subject of a given certificate.
    ///
    func emailAddresses() throws -> [String] { }

    /// Return the certificate's normalized issuer
    ///
    func normalizedIssuerSequence() -> Data { }

    /// Return the certificate's normalized subject
    ///
    func normalizedSubjectSequence() -> Data { }

    /// Retrieves the public key for a given certificate.
    ///
    func publicKey() throws -> Data { }

    /// Return the certificate's serial number.
    ///
    func serialNumberData() throws -> Data { }

    /// Returns the private key associated with an identity.
    ///
    func copyPrivateKey() throws -> Data {}
 }
