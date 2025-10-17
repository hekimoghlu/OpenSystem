/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
import XCTest
import WebKit

class WebKitTests: XCTestCase {
    /// This is a compile-time test that ensures the function names are what we expect.
    func testAPI() {
        _ = WKContentWorld.world(name:)
        _ = WKWebView.callAsyncJavaScript(_:arguments:in:in:completionHandler:)
        _ = WKWebView.createPDF(configuration:completionHandler:)
        _ = WKWebView.createWebArchiveData(completionHandler:)
        _ = WKWebView.evaluateJavaScript(_:in:in:completionHandler:)
        _ = WKWebView.find(_:configuration:completionHandler:)
    }

    func testWKPDFConfigurationRect() {
        let configuration = WKPDFConfiguration()

        XCTAssert(type(of: configuration.rect) == Optional<CGRect>.self)

        configuration.rect = nil
        XCTAssertEqual(configuration.rect, nil)

        configuration.rect = .null
        XCTAssertEqual(configuration.rect, nil)

        configuration.rect = CGRect.zero
        XCTAssertEqual(configuration.rect, .zero)

        let originalPhoneBounds = CGRect(x: 0, y: 0, width: 320, height: 480)
        configuration.rect = originalPhoneBounds
        XCTAssertEqual(configuration.rect, originalPhoneBounds)
    }
}
