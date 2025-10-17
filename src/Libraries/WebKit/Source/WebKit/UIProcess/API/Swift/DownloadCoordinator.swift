/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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

// Copyright (C) 2024 Apple Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY APPLE INC. AND ITS CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL APPLE INC. OR ITS CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

#if ENABLE_SWIFTUI && compiler(>=6.0)

import Foundation

// MARK: DownloadCoordinator protocol

@_spi(Private)
public protocol DownloadCoordinator {
    @MainActor
    func destination(forDownload download: WebPage_v0.DownloadID, response: URLResponse, suggestedFilename: String) async -> URL?

    @MainActor
    func authenticationChallengeDisposition(forDownload download: WebPage_v0.DownloadID, challenge: URLAuthenticationChallenge) async -> (URLSession.AuthChallengeDisposition, URLCredential?)

    @MainActor
    func httpRedirectionPolicy(forDownload download: WebPage_v0.DownloadID, response: HTTPURLResponse, newRequest request: URLRequest) async -> WebPage_v0.Download.RedirectPolicy

    @MainActor
    func placeholderPolicy(forDownload download: WebPage_v0.DownloadID) async -> WebPage_v0.Download.PlaceholderPolicy
}

// MARK: Default implementation

@_spi(Private)
public extension DownloadCoordinator {
    @MainActor
    func destination(forDownload download: WebPage_v0.DownloadID, response: URLResponse, suggestedFilename: String) async -> URL? {
        nil
    }

    @MainActor
    func authenticationChallengeDisposition(forDownload download: WebPage_v0.DownloadID, challenge: URLAuthenticationChallenge) async -> (URLSession.AuthChallengeDisposition, URLCredential?) {
        (.performDefaultHandling, nil)
    }

    @MainActor
    func httpRedirectionPolicy(forDownload download: WebPage_v0.DownloadID, response: HTTPURLResponse, newRequest request: URLRequest) async -> WebPage_v0.Download.RedirectPolicy {
        .allow
    }

    @MainActor
    func placeholderPolicy(forDownload download: WebPage_v0.DownloadID) async -> WebPage_v0.Download.PlaceholderPolicy {
        .disable(alternatePlaceholder: nil)
    }
}

#endif
