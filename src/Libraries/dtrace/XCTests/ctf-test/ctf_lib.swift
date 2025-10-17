/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
//  ctf_lib.swift
//  dtrace
//
//  Created by tjedlicka on 5/11/22.
//

import XCTest

class ctf_lib: XCTestCase {

	func testVersion() {
		XCTAssertEqual(CTF_VERSION, CTF_VERSION_4)
	}

	func testContainerCreate() {
		var ctf_container: UnsafeMutablePointer<ctf_file_t>?
		var err: Int32 = 0

		ctf_container = ctf_create(&err)

		XCTAssertNotNil(ctf_container)
		XCTAssertEqual(err, 0)

		ctf_close(ctf_container)
	}
}
