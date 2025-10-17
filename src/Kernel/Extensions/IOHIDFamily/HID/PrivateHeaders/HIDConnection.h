/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
#ifndef HIDConnection_h
#define HIDConnection_h

#import <Foundation/Foundation.h>
#import <HID/HIDBase.h>
#import <IOKit/hidobjc/HIDConnectionBase.h>

NS_ASSUME_NONNULL_BEGIN

/*!
 * @category HIDConnection
 *
 * @abstract
 * Direct interaction with a HID event system connection.
 *
 * @discussion
 * Should only be used by system code.
 */
@interface HIDConnection (HIDFramework)

@property (readonly) NSString *uuid;

-(void)getAuditToken:(audit_token_t *)token;

@end

NS_ASSUME_NONNULL_END

#endif /* HIDConnection_h */
