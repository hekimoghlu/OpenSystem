/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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

#import "internal.h"

#import <XCTest/XCTest.h>

#define XCTAssertNotNull(ptr) XCTAssertNotEqual(ptr, NULL)

@interface magazine_small_tests : XCTestCase {
@private
	struct rack_s small_rack;
}
@end

@implementation magazine_small_tests

- (void)setUp {
	memset(&small_rack, 'a', sizeof(small_rack));
	rack_init(&small_rack, RACK_TYPE_SMALL, 1, 0);
}

- (void *)small_malloc:(size_t)size {
	return small_malloc_should_clear(&small_rack, SMALL_MSIZE_FOR_BYTES(size), false);
}

- (void)testSmallMallocSucceeds {
	XCTAssertNotNull([self small_malloc:512]);
}

- (void)testSmallRegionFoundAfterMalloc {
	void *ptr = [self small_malloc:512];
	XCTAssertNotNull(ptr);

	XCTAssertNotNull(small_region_for_ptr_no_lock(&small_rack, ptr));
}

- (void)testSmallSizeMatchesMalloc {
	void *ptr = [self small_malloc:512];
	XCTAssertNotNull(ptr);

	XCTAssertEqual(small_size(&small_rack, ptr), 512);
}

@end
