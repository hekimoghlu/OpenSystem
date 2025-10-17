/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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

#if OCTAGON

class OctagonErrorUtilsTest: OctagonTestsBase {
    func testAKErrorRetryable() throws {
        let urlError = NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut, userInfo: nil)
        let error = NSError(domain: AKAppleIDAuthenticationErrorDomain, code: 17, userInfo: [NSUnderlyingErrorKey: urlError])
        print(error)
        XCTAssertTrue(error.isRetryable(), "AK/NSURLErrorTimedOut should be retryable")
    }

    func testURLErrorRetryable() throws {
        let error = NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut, userInfo: nil)
        print(error)
        XCTAssertTrue(error.isRetryable(), "NSURLErrorTimedOut should be retryable")
    }

    func testNotConnectedRetryable() throws {
        let error = NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet, userInfo: nil)
        print(error)
        XCTAssertTrue(error.isRetryable(), "NSURLErrorNotConnectedToInternet should be retryable")
    }

    func testCKErrorRetryable() throws {
        let urlError = NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet, userInfo: nil)
        let ckError = NSError(domain: CKErrorDomain, code: CKError.networkUnavailable.rawValue, userInfo: [NSUnderlyingErrorKey: urlError])
        print(ckError)
        XCTAssertTrue(ckError.isRetryable(), "CK/NSURLErrorNotConnectedToInternet should be retryable")
    }

    func testRetryIntervalCKError() throws {
        let error = NSError(domain: CKErrorDomain, code: 17, userInfo: nil)
        print(error)
        XCTAssertEqual(2, error.retryInterval(), "expect CKError default retry to 2")
    }

    func testRetryIntervalCKErrorRetry() throws {
        let error = NSError(domain: CKErrorDomain, code: 17, userInfo: [CKErrorRetryAfterKey: 17])
        print(error)
        XCTAssertEqual(17, error.retryInterval(), "expect CKError default retry to 17")
    }

    func testRetryIntervalCKErrorRetryBad() throws {
        let error = NSError(domain: CKErrorDomain, code: 17, userInfo: [CKErrorRetryAfterKey: "foo"])
        print(error)
        XCTAssertEqual(2, error.retryInterval(), "expect CKError default retry to 2")
    }

    func testRetryIntervalCKErrorPartial() throws {
        let suberror = NSError(domain: CKErrorDomain, code: 1, userInfo: [CKErrorRetryAfterKey: "4711"])

        let error = NSError(domain: CKErrorDomain, code: CKError.partialFailure.rawValue, userInfo: [CKPartialErrorsByItemIDKey: ["foo": suberror]])
        print(error)
        XCTAssertEqual(4711, error.retryInterval(), "expect CKError default retry to 4711")
    }

    func testRetryIntervalCuttlefish() throws {
        let cuttlefishError = NSError(domain: CuttlefishErrorDomain,
                                      code: 17,
                                      userInfo: [CuttlefishErrorRetryAfterKey: 101])
        let internalError = NSError(domain: CKUnderlyingErrorDomain,
                                    code: CKUnderlyingError.pluginError.rawValue,
                                    userInfo: [NSUnderlyingErrorKey: cuttlefishError, ])
        let ckError = NSError(domain: CKErrorDomain,
                              code: CKError.serverRejectedRequest.rawValue,
                              userInfo: [NSUnderlyingErrorKey: internalError,
                                  CKErrorServerDescriptionKey: "Fake: FunctionError domain: CuttlefishError, 17",
                                        ])
        print(ckError)
        XCTAssertEqual(101, ckError.retryInterval(), "cuttlefish retry should be 101")
    }

    func testCuttlefishRetryAfter() throws {
        let error = NSError(domain: CKErrorDomain, code: 17, userInfo: nil)
        print(error)
        XCTAssertEqual(2, error.retryInterval(), "expect default retry of 2")
        XCTAssertEqual(0, error.cuttlefishRetryAfter(), "expect cuttlefish retry of 0")
    }
}

#endif
