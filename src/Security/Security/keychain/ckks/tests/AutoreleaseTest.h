/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#define TEST_API_AUTORELEASE_BEFORE(name) size_t _pending_before_##name = pending_autorelease_count()

#define TEST_API_AUTORELEASE_AFTER(name)                        \
    size_t _pending_after_##name = pending_autorelease_count(); \
    XCTAssertEqual(_pending_before_##name, _pending_after_##name, "pending autoreleases unchanged (%lu->%lu)", _pending_before_##name, _pending_after_##name)

ssize_t pending_autorelease_count(void);
