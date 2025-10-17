/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
internal import WebKit_Internal

@MainActor
@_spi(Private)
public struct URLScheme_v0: Hashable, Sendable {
    public init?(_ rawValue: String) {
        guard WKWebViewConfiguration._isValidCustomScheme(rawValue) else {
            return nil
        }

        self.rawValue = rawValue
    }

    let rawValue: String
}

@_spi(Private)
public enum URLSchemeTaskResult_v0: Sendable {
    case response(URLResponse)

    case data(Data)
}

@_spi(Private)
public protocol URLSchemeHandler_v0 {
    associatedtype TaskSequence: AsyncSequence<URLSchemeTaskResult_v0, any Error>

    func reply(for request: URLRequest) -> TaskSequence
}

// MARK: Adapters

final class WKURLSchemeHandlerAdapter: NSObject, WKURLSchemeHandler {
    init(_ wrapped: any URLSchemeHandler_v0) {
        self.wrapped = wrapped
    }

    private let wrapped: any URLSchemeHandler_v0

    private var tasks: [ObjectIdentifier: Task<Void, Never>] = [:]

    func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
        let task = Task {
            do {
                for try await result in wrapped.reply(for: urlSchemeTask.request) {
                    switch result {
                    case let .response(response):
                        urlSchemeTask.didReceive(response)

                    case let .data(data):
                        urlSchemeTask.didReceive(data)
                    }
                }

                urlSchemeTask.didFinish()
            } catch {
                urlSchemeTask.didFailWithError(error)
            }
        }

        tasks[ObjectIdentifier(urlSchemeTask)] = task
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) {
        tasks.removeValue(forKey: ObjectIdentifier(urlSchemeTask))?.cancel()
    }
}

#endif
