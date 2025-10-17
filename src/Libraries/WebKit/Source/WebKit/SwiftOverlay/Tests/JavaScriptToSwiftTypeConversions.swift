/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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

final class JavaScriptToSwiftConversions : XCTestCase {
    #if os(macOS)
    let window = NSWindow()
    #elseif os(iOS)
    let window = UIWindow()
    #endif

    let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 320, height: 480))

    override func setUp() {
        #if os(macOS)
        window.contentView?.addSubview(webView)
        #else
        window.isHidden = false
        window.addSubview(webView)
        #endif

        webView.load(URLRequest(url: URL(string: "about:blank")!))
    }

    override func tearDown() {
        #if os(macOS)
        window.orderOut(nil)
        #else
        window.isHidden = true
        #endif
    }

    func evaluateJavaScript<T : Equatable>(_ javaScript: String, andExpect expectedValue: T) {
        let evaluationExpectation = self.expectation(description: "Evaluation of \(javaScript.debugDescription)")
        webView.evaluateJavaScript(javaScript, in: nil, in: .defaultClient) { result in
            do {
                let actualValue = try result.get() as? T
                XCTAssertEqual(actualValue, expectedValue)
                evaluationExpectation.fulfill()
            } catch {
                XCTFail("Evaluating \(javaScript.debugDescription) failed with error: \(error)")
            }
        }

        wait(for: [evaluationExpectation], timeout: 30)
    }

    func testNull() {
        evaluateJavaScript("null", andExpect: NSNull())
    }

    func testInteger() {
        evaluateJavaScript("12", andExpect: 12 as Int)
    }

    func testDecimal() {
        evaluateJavaScript("12.0", andExpect: 12 as Float)
    }

    func testBoolean() {
        evaluateJavaScript("true", andExpect: true)
        evaluateJavaScript("false", andExpect: false)
    }

    func testString() {
        evaluateJavaScript(#""Hello, world!""#, andExpect: "Hello, world!")
    }

    func testArray() {
        // This uses [AnyHashable], instead of [Any], so we can perform an equality check for testing.
        evaluateJavaScript(#"[ 1, 2, "cat" ]"#, andExpect: [1, 2, "cat"] as [AnyHashable])
    }

    func testDictionary() {
        // This uses [AnyHashable:AnyHashable], instead of [AnyHashable:Any], so we can perform an
        // equality check for testing. An objectâ€™s keys are always converted to strings, so even
        // though we input `1:` we expect `"1":` back.
        let result: [AnyHashable:AnyHashable] = ["1": 2, "cat": "dog"]
        evaluateJavaScript(#"const value = { 1: 2, "cat": "dog" }; value"#, andExpect: result)
    }

    func testUndefined() {
        let evaluationExpectation = self.expectation(description: "Evaluation of \"undefined\" using deprecated API")
        webView.evaluateJavaScript("undefined", in: nil, in: .defaultClient) { (result: Result<Any, Error>) in
            do {
                let value = try result.get()
                if let optionalValue = value as? Any?, optionalValue == nil {
                    evaluationExpectation.fulfill()
                } else {
                    XCTFail("Value did not contain nil")
                }
            } catch {
                XCTFail("Evaluating \"undefined\" failed with error: \(error)")
            }
        }

        wait(for: [evaluationExpectation], timeout: 30)
    }

    #if swift(>=5.5)
    @available(iOS 15.0, macOS 12.0, *)
    func testUsingSwiftAsync() async throws {
        guard let result = try await webView.evaluateJavaScript(#""Hello, world!""#) as? String else {
            XCTFail("Unexpected result from evaluating JavaScript.")
            return
        }

        XCTAssertEqual(result, "Hello, world!")
    }
    #endif
}
