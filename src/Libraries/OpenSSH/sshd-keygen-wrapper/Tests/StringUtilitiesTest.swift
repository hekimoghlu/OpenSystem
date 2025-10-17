/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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

final class StringUtilitiesTest: XCTestCase {
  func testEnvironmentVariableNotPresent() throws {
    XCTAssertNil(
      String(fromEnvironmentVariable: "No such variable"),
      "Expected nil for invalid environment variable name.")
  }

  func testEnvironmentalVariablePresent() throws {
    setenv("SSHDWRAPPERTESTVAR", "forty-two", 1)
    addTeardownBlock {
      unsetenv("SSHDWRAPPERTESTVAR")
    }
    XCTAssertEqual(String(fromEnvironmentVariable: "SSHDWRAPPERTESTVAR"), "forty-two")
  }

  func testQuoteNothingSpecial() throws {
    let s = "No-unsafe_characters@here,dudes+%:./42"
    XCTAssertEqual(s, s.shellQuoted)
  }

  func testQuoteSpaces() throws {
    let s = "This has spaces in it."
    XCTAssertEqual(s.shellQuoted, "'\(s)'")
  }

  func testQuoteSpecial() throws {
    let s = "This has a dollar sign in it: $HOME."
    XCTAssertEqual(s.shellQuoted, "'\(s)'")
  }

  func testEmbeddedQuotes() throws {
    let s = "This contains 'single quotes' as well as \"double quotes\"."
    XCTAssertEqual(
      s.shellQuoted, "'This contains '\\''single quotes'\\'' as well as \"double quotes\".'")
  }

  func testQuoteEmptyString() throws {
    let s = ""
    XCTAssertEqual(s.shellQuoted, "''")
  }
}
