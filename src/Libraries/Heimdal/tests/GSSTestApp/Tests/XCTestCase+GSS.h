/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
//  SenTestCase+GSS.h
//  GSSTestApp
//
//  Created by Love HÃ¶rnquist Ã…strand on 2013-06-08.
//  Copyright (c) 2013 Apple, Inc. All rights reserved.
//

#import "FakeXCTest.h"
#include <GSS/GSS.h>

@interface XCTestCase (GSS)

- (void)XTCDestroyCredential:(gss_OID)mech;
- (gss_cred_id_t)XTCAcquireCredential:(NSString *)name withOptions:(NSDictionary *)options mech:(gss_OID)mech;
- (BOOL)STCAuthenticate:(gss_cred_id_t)cred nameType:(gss_OID)nameType toServer:(NSString *)serverName;
- (void)XCTOutput:(NSString *)output, ... NS_FORMAT_FUNCTION(1,2);

@end
