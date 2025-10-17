/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
//  SecProtocolHelperTest.m
//  SecProtocol
//

#import <XCTest/XCTest.h>

#import "SecProtocolInternal.h"

#define DefineTLSCiphersuiteGroupList(XXX, ...) \
    static const tls_ciphersuite_t list_##XXX[] = { \
        __VA_ARGS__ \
    };

// Mirror the internal definition of this ciphersuite group
DefineTLSCiphersuiteGroupList(tls_ciphersuite_group_default, CiphersuitesTLS13, CiphersuitesPFS);

#undef DefineTLSCiphersuiteGroupList

@interface SecProtocolHelperTest : XCTestCase
@end

@implementation SecProtocolHelperTest

- (void)testCiphersuiteGroupConversion {
    size_t ciphersuites_len = 0;
    const tls_ciphersuite_t *ciphersuites = sec_protocol_helper_ciphersuite_group_to_ciphersuite_list(tls_ciphersuite_group_default, &ciphersuites_len);
    XCTAssertTrue(ciphersuites != NULL);
    XCTAssertTrue(ciphersuites_len == (sizeof(list_tls_ciphersuite_group_default) / sizeof(tls_ciphersuite_t)));
    for (size_t i = 0; i < ciphersuites_len; i++) {
        XCTAssertTrue(ciphersuites[i] == list_tls_ciphersuite_group_default[i]);
    }
}

@end
